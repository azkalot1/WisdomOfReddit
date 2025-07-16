# batch_process_sentiment_range.py
import asyncio, json, math, time, yaml, logging
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

from openai import OpenAI

# ── internal imports (identical to your async script) ─────────────────
from internetwisdom.analysis.sentiment import (
    LocalFileContentLoader, IncrementalSentimentStorage,  # still used
    AsyncSentimentProcessor,                              # only for .extractor
    ModelConfig as SentimentModelConfig,                  # alias
    RelevancePrescorer, SentimentDeduplicator,
    DeduplicationConfig as SentimentDeduplicationConfig,
    DeduplicationStrategy, ConflictResolution,
    TickerSentimentAggregator, HybridConsensusStorage,
    SentimentAnalysisConfigWithBatch,
    RangeBatchOrchestrator, BatchOutputParser,
    DateIndexStore
)
from internetwisdom.analysis.prescoring import (
    create_default_featurizer, RelevancePredictor
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# =====================================================================
class BatchSentimentProcessor:
    """
    Submit at most <max_active_jobs> Batch-API jobs, poll until done,
    parse each output into DailySentimentBatch, then dedupe→aggregate→store.
    """

    def __init__(self, cfg: SentimentAnalysisConfigWithBatch):
        self.cfg = cfg
        self._init_components()

    # -----------------------------------------------------------------
    def _init_components(self):
        c = self.cfg
        # loaders / prescorer (exactly like async version)
        self.loader = LocalFileContentLoader(
            Path(c.paths.reddit_comments_path),
            max_n_comments=c.processing_config.max_n_comments
        )
        featurizer = create_default_featurizer()
        pred = RelevancePredictor.load_from_folder(
            c.paths.relevance_model_path, featurizer
        )
        self.prescorer = RelevancePrescorer(
            pred, threshold=c.processing_config.relevance_threshold
        )

        # sentiment_processor only to access .extractor & system prompt
        self.sent_proc = AsyncSentimentProcessor(
            model_config=SentimentModelConfig(**vars(c.model_config)),
            async_config=None
        )

        # range orchestrator & result parser
        self.rbo = RangeBatchOrchestrator(
            content_loader      = self.loader,
            sentiment_processor = self.sent_proc,
            prescorer           = self.prescorer,
            tmp_dir             = Path(c.paths.batch_jobs_path)
        )
        self.parser = BatchOutputParser(
            tmp_dir=Path(c.paths.batch_outputs_path) 
        )

        # downstream helpers
        dedup_cfg = SentimentDeduplicationConfig(**vars(c.deduplication_config))
        self.deduplicator = SentimentDeduplicator(config=dedup_cfg)
        self.aggregator   = TickerSentimentAggregator()
        self.cons_storage = HybridConsensusStorage(
            local_path=Path(c.paths.consensus_data_path),
            s3_bucket=c.processing_config.s3_bucket,
            s3_prefix=c.paths.s3_prefix,
            local_retention_days=c.processing_config.local_retention_days
        )

        self.date_index = DateIndexStore(
            s3     = self.cons_storage.s3_storage.s3_client,
            bucket = self.cfg.processing_config.s3_bucket,
            key    = f"{self.cfg.paths.s3_prefix}/metadata/date_index.json"
        )

        # index of submitted jobs (survives restarts)
        self.index_path = Path(c.paths.batch_jobs_path) / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text("[]")

        self.client = OpenAI()

    # ------------------------ utility --------------------------------
    def _load_index(self) -> List[Dict]:
        return json.loads(self.index_path.read_text())

    def _save_index(self, data: List[Dict]):
        self.index_path.write_text(json.dumps(data, indent=2))

    def _generate_dates(self) -> List[str]:
        dr = self.cfg.date_range
        s = datetime.strptime(dr.start_date, "%Y%m%d")
        e = datetime.strptime(dr.end_date, "%Y%m%d")
        return [(s + timedelta(days=i)).strftime("%Y%m%d")
                for i in range((e - s).days + 1)]

    def _submit_single_day(self, date: str):
        processed = self.date_index.processed_dates()
        if date in processed:
            return

        idx = self._load_index()
        if any(date in r["dates"] and r["state"] == "pending" for r in idx):
            return  # already queued
        logger.info(f"Submitting {date}") 
        recs = self.rbo.submit_range(
            date, date,
            submit_per_day=True,
            completion_window="24h"
        )
        for r in recs:
            r["state"] = "pending"
        idx.extend(recs)
        self._save_index(idx)

    # ------------------ submission logic -----------------------------
    def _submit_more(self):
        idx = self._load_index()
        active = [j for j in idx if j["state"] == "pending"]
        need   = self.cfg.batch.max_active_jobs - len(active)
        if need <= 0:
            return

        done = self.date_index.processed_dates()
        submitted = {d for j in idx for d in j["dates"]}

        for date in self._generate_dates():
            if need == 0:
                break
            if date in submitted or date in done:
                continue

            self._submit_single_day(date)
            need -= 1

    # --------------------- polling logic -----------------------------
    async def _poll_and_ingest(self):
        idx      = self._load_index()
        retry    : set[str] = set()
        changed  = False

        for rec in idx[:]:                       # iterate over a copy
            if rec["state"] != "pending":
                continue

            job = self.client.batches.retrieve(rec["job_id"])

            # -------- completed ---------------------------------------
            if job.status == "completed":
                for batch in self.parser.parse_job(rec["job_id"]).values():
                    deduped = [
                        self.deduplicator.deduplicate(v)
                        for v in batch.submission_sentiments.values()
                    ]
                    cons = self.aggregator.aggregate_daily_sentiments(
                        deduped, batch.date
                    )
                    self.cons_storage.save_daily_consensus(cons)

                rec["state"] = "completed"
                changed = True
                logger.info(f"✓ ingested {rec['job_id']}")

            # -------- failed ------------------------------------------
            elif job.status == "failed":
                err = getattr(job, "errors", None) or getattr(job, "error", None)
                try:
                    err = err.model_dump() if err else "unknown"
                except AttributeError:
                    err = err.dict() if err else "unknown"

                retry.update(rec["dates"])      # queue for resubmit
                idx.remove(rec)                 # drop failed record
                changed = True
                logger.error(f"✗ {rec['job_id']} failed: {err}")

        if changed:
            self._save_index(idx)

        # -------- resubmit any dates from failed jobs ------------------
        for d in sorted(retry):
            logger.info(f"Resubmitting {d}")
            self._submit_single_day(d)

    # --------------------- main driver -------------------------------
    async def run(self):
        """
        Main driver: keep ≤ max_active_jobs pending, poll finished jobs,
        ingest results, and exit as soon as every date in the configured
        range is marked processed.
        """
        all_dates = set(self._generate_dates())          # calendar span

        # ── pre-flight shortcut ───────────────────────────────────────────
        if all_dates.issubset(self.date_index.processed_dates()) and not self.cfg.flags.force_overwrite:
            logger.info("Nothing to do: entire date range already processed.")
            return

        logger.info("Batch sentiment processing started")
        poll_s = self.cfg.batch.poll_seconds

        while True:
            # 1) keep the queue full (skip already-processed dates)
            self._submit_more()

            # 2) poll finished jobs, ingest & mark dates done
            await self._poll_and_ingest()

            # 3) evaluate exit condition
            processed = self.date_index.processed_dates()
            idx       = self._load_index()
            pending   = sum(j["state"] == "pending" for j in idx)

            remaining_dates = [
                d for d in all_dates
                if d not in processed                         # not done
                and d not in {dd for j in idx for dd in j["dates"]}  # not pending
            ]

            if pending == 0 and not remaining_dates:
                logger.info("🏁  All dates processed. Exiting main loop.")
                break

            await asyncio.sleep(poll_s)


# =====================================================================
# -------- YAML config loader identical to original -------------------
def _dict_to_dc(cls, data):
    fld = {f.name: f.type for f in fields(cls)}
    return cls(**{k: _dict_to_dc(fld[k], v) if hasattr(fld[k], "__dataclass_fields__") else v
                  for k, v in data.items()})

def load_cfg(path: str) -> SentimentAnalysisConfigWithBatch:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # enum conversion for dedup strategy / conflict
    ded = raw["deduplication_config"]
    ded["strategy"]            = DeduplicationStrategy[ded["strategy"]]
    ded["conflict_resolution"] = ConflictResolution[ded["conflict_resolution"]]
    return _dict_to_dc(SentimentAnalysisConfigWithBatch, raw)

# =====================================================================
async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    proc = BatchSentimentProcessor(cfg)
    await proc.run()

if __name__ == "__main__":
    asyncio.run(main())
