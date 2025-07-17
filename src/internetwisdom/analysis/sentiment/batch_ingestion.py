from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path
import json, re
from typing import Dict, List, Tuple

from openai import OpenAI
from pydantic import ValidationError

# domain objects you already have
from .state     import SentimentResults, DailySentimentBatch
from .deduplication import SentimentDeduplicator
from .aggregation   import TickerSentimentAggregator
from .storing     import SentimentStorage

_CID_RE = re.compile(r"^(?P<date>\d{8})_(?P<sid>[^_]+)_c\d+$")

class BatchOutputParser:
    """
    Pure-parsing helper:
        job_id      → download output file
                    → parse every line into SentimentResults
                    → regroup by (date, submission_id)
                    → build DailySentimentBatch for each <date>
        returns {date: DailySentimentBatch}
    """

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        tmp_dir: Path = Path("./batch_outputs")
    ) -> None:
        self.client  = client or OpenAI()
        self.tmp_dir = tmp_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def load_by_job_path(self, job_path: Path, job_id: str = 'missing') -> Dict[str, DailySentimentBatch]:
        bucket: Dict[Tuple[str, str], List[SentimentResults]] = defaultdict(list)

        with job_path.open() as fh:
            for line in fh:
                obj = json.loads(line)
                cid = obj["custom_id"]

                m = _CID_RE.match(cid)
                if not m:
                    continue  # malformed custom_id

                date, sid = m["date"], m["sid"]

                body = obj.get("response", {}).get("body", {})
                if "error" in body:
                    continue  # skip failed calls

                try:
                    arg_str = (
                        body["choices"][0]["message"]["tool_calls"][0]
                            ["function"]["arguments"]
                    )
                    parsed = json.loads(arg_str)
                    sr = SentimentResults(**parsed)
                    bucket[(date, sid)].append(sr)
                except (KeyError, json.JSONDecodeError, ValidationError):
                    # silently skip malformed rows
                    continue

        # -------- build DailySentimentBatch objects ------------------
        by_date: Dict[str, Dict[str, List[SentimentResults]]] = defaultdict(dict)
        for (date, sid), extractions in bucket.items():
            by_date[date][sid] = extractions

        daily_batches: Dict[str, DailySentimentBatch] = {}
        for date, sub_map in by_date.items():
            total_extract = sum(len(v) for v in sub_map.values())
            daily_batches[date] = DailySentimentBatch(
                date                 = date,
                submission_sentiments= sub_map,
                processing_metadata  = {"source_job_id": job_id},
                last_updated         = datetime.utcnow(),
                total_submissions    = len(sub_map),
                total_extractions    = total_extract
            )

        return daily_batches

    # -----------------------------------------------------------------
    def parse_job(self, job_id: str) -> Dict[str, DailySentimentBatch]:
        job = self.client.batches.retrieve(job_id)
        if job.status != "completed":
            raise RuntimeError(f"job {job_id} not finished (status={job.status})")

        output_path = self._download_output(job.output_file_id)

        return self.load_by_job_path(output_path, job_id)

    # -----------------------------------------------------------------
    def _download_output(self, file_id: str) -> Path:
        raw = self.client.files.content(file_id).content
        path = self.tmp_dir / f"{file_id}.jsonl"
        path.write_bytes(raw)
        return path
