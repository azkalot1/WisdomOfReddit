from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Iterable, Optional
from .loading import ContentLoader
from .processing import SentimentProcessor
from .storing import SentimentStorage, IncrementalSentimentStorage
from .state import DailyUpdate, DailySentimentBatch, ProcessingCheckpoint
from datetime import datetime
from abc import ABC, abstractmethod
from .prescoring import SentimentPrescorer
from openai import OpenAI
import logging

# SETUP LOGGING
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------
# helper dataclass for metadata (simple, no pydantic needed here)
# ---------------------------------------------------------------------
class BatchJobRecord(dict):
    """
    Minimal JSON-serialisable record you can stash on disk/S3 to
    reconstruct how a batch was built.
    """

    @classmethod
    def new(cls, *, job_id: str, file_id: str,
            dates: List[str], request_count: int) -> "BatchJobRecord":
        return cls(
            job_id=job_id,
            file_id=file_id,
            dates=dates,
            request_count=request_count,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
            custom_id_format="<date>_<submission_id>_c<idx>"
        )


# ---------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------
class RangeBatchOrchestrator:
    """
    Bundle many calendar days into as few Batch-API jobs as necessary.

    Parameters
    ----------
    content_loader       : your implementation of ContentLoader
    sentiment_processor  : instance whose `extractor` formats prompts
    prescorer            : optional prescorer (same you use now)
    tmp_dir              : where *.jsonl and metadata files live
    max_requests_per_job : "soft" guard < 50_000
    max_jsonl_bytes      : "soft" guard < 200*1024*1024  (OPENAI limit)
    """
    def __init__(
        self,
        *,
        content_loader: ContentLoader,
        sentiment_processor: SentimentProcessor,
        prescorer: SentimentPrescorer,
        tmp_dir: Path = Path("./tmp_batch_jobs"),
        max_requests_per_job: int = 40_000,
        max_jsonl_bytes: int = 180 * 1024 * 1024
    ) -> None:
        self.loader = content_loader
        self.processor = sentiment_processor      # provides .extractor
        self.prescorer = prescorer
        self.tmp_dir = tmp_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.max_requests = max_requests_per_job
        self.max_bytes    = max_jsonl_bytes
        self.client       = OpenAI()

        # cache for schema / system prompt
        self.schema = sentiment_processor.extractor.output_model.model_json_schema()
        self.system_prompt = sentiment_processor.extractor.system_prompt

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def submit_range(
        self,
        start_date: str,          # "YYYYMMDD"
        end_date:   str,          # "YYYYMMDD"
        *,
        completion_window: str = "24h",
        submit_per_day: bool = False          # <- NEW FLAG
    ) -> List[BatchJobRecord]:
        """
        Build one or many Batch-API jobs covering <start_date,end_date>.

        If `submit_per_day=True`, force-flush at day boundaries so that every
        calendar day ends up in its own Batch job (unless you also hit the
        size / request limits sooner).

        Returns
        -------
        List[BatchJobRecord]   # one element per submitted job
        """
        all_dates = sorted(
            d for d in self.loader.get_available_dates()
            if start_date <= d <= end_date
        )

        job_records: List[BatchJobRecord] = []
        current_tasks: List[Dict] = []
        current_dates: List[str]  = []
        current_bytes: int = 0

        def flush() -> None:
            nonlocal current_bytes
            if not current_tasks:
                return
            rec = self._submit_job(
                tasks=current_tasks,
                dates=current_dates,
                completion_window=completion_window
            )
            job_records.append(rec)
            current_tasks.clear()
            current_dates.clear()
            current_bytes = 0

        # ---------------- iterate calendar --------------------------------
        for date in all_dates:
            sub_ids = self.loader.get_submissions_for_date(date)
            for sub_id in sub_ids:
                content = self.loader.load_submission_content(date, sub_id)
                if not content:
                    continue
                if self.prescorer and not self.prescorer.predict(content):
                    logger.info(f"Skipping {sub_id} because it is not relevant")
                    continue

                for idx, comment in enumerate(content.comments):
                    user_msg = self.processor.extractor._format_single_extraction(
                        title=content.title,
                        submission_body=content.submission_body,
                        comment_text=comment
                    )
                    cid  = f"{date}_{sub_id}_c{idx}"
                    task = self._make_task(cid, user_msg)
                    t_sz = len(json.dumps(task).encode())

                    hit_limit = (
                        (len(current_tasks) + 1 > self.max_requests) or
                        (current_bytes + t_sz > self.max_bytes)
                    )
                    if hit_limit:
                        flush()

                    current_tasks.append(task)
                    current_dates.append(date)
                    current_bytes += t_sz

            # ---- end-of-day forced flush? --------------------------------
            if submit_per_day:
                flush()

        # any leftovers
        flush()
        return job_records

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    def _make_task(self, custom_id: str, user_prompt: str) -> Dict:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.processor.extractor.model_config.model_name,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "SentimentResults",
                        "description": "Return structured sentiment info.",
                        "parameters": self.schema
                    }
                }],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "SentimentResults"}
                }
            }
        }

    def _submit_job(self, *, tasks: List[Dict],
                    dates: List[str],
                    completion_window: str) -> BatchJobRecord:

        # 1. write JSONL
        if len(dates) == 1:
            span = dates[0]                      # "20200101"
        else:
            span = f"{dates[0]}_{dates[-1]}"     # "20200101_20200107"
        jsonl_path = self.tmp_dir / f"{span}_{len(tasks)}req.jsonl"
        with jsonl_path.open("w") as fh:
            for t in tasks:
                fh.write(json.dumps(t) + "\n")

        # 2. upload + create batch
        file_res = self.client.files.create(file=jsonl_path.open("rb"),
                                            purpose="batch")
        job_res = self.client.batches.create(
            input_file_id=file_res.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window
        )

        # 3. log metadata
        record = BatchJobRecord.new(
            job_id=job_res.id,
            file_id=file_res.id,
            dates=sorted(set(dates)),
            request_count=len(tasks)
        )
        meta_path = jsonl_path.with_suffix(".metadata.json")
        meta_path.write_text(json.dumps(record, indent=2))

        print(f"🚀  submitted {len(tasks)} requests "
              f"[{span}] → job {job_res.id}")
        return record
