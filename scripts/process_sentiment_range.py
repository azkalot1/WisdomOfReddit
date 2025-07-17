# process_sentiment_range.py
import asyncio
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import logging

from internetwisdom.analysis.sentiment import (
    LocalFileContentLoader,
    IncrementalSentimentStorage,
    IncrementalSentimentOrchestrator,
    AsyncExtractionConfig,
    ModelConfig as SentimentModelConfig,
    AsyncSentimentProcessor,
    RelevancePrescorer,
    SentimentDeduplicator,
    DeduplicationConfig as SentimentDeduplicationConfig,
    DeduplicationStrategy, 
    ConflictResolution,
    TickerSentimentAggregator,
    HybridConsensusStorage,
    SentimentAnalysisConfig,
    DateIndexStore
)
from internetwisdom.analysis.prescoring import (
    create_default_featurizer,
    RelevancePredictor,
)
import dotenv
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentProcessor:
    def __init__(self, config: SentimentAnalysisConfig):
        self.config = config
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all processing components based on configuration."""
        # Content loader
        self.content_loader = LocalFileContentLoader(
            Path(self.config.paths.reddit_comments_path), 
            max_n_comments=self.config.processing_config.max_n_comments
        )
        
        # Sentiment storage
        self.sentiment_storage = IncrementalSentimentStorage(
            Path(self.config.paths.processed_sentiments_path)
        )
        
        # Async configuration
        async_config = AsyncExtractionConfig(
            max_concurrent_requests=self.config.async_config.max_concurrent_requests,
            requests_per_minute=self.config.async_config.requests_per_minute,
            tokens_per_minute=self.config.async_config.tokens_per_minute,
            batch_size=self.config.async_config.batch_size,
            delay_between_batches=self.config.async_config.delay_between_batches,
            enable_rate_limiting=self.config.async_config.enable_rate_limiting,
            token_buffer_ratio=self.config.async_config.token_buffer_ratio,
            request_buffer_ratio=self.config.async_config.request_buffer_ratio
        )
        
        # Relevance prescorer
        featurizer = create_default_featurizer()
        predictor = RelevancePredictor.load_from_folder(
            self.config.paths.relevance_model_path,
            featurizer
        )
        self.prescorer = RelevancePrescorer(
            predictor, 
            threshold=self.config.processing_config.relevance_threshold
        )
        
        # Model configuration
        model_config = SentimentModelConfig(
            model_name=self.config.model_config.model_name,
            temperature=self.config.model_config.temperature,
            max_tokens=self.config.model_config.max_tokens,
            timeout=self.config.model_config.timeout,
            max_retries=self.config.model_config.max_retries
        )
        
        # Sentiment processor
        self.sentiment_processor = AsyncSentimentProcessor(
            model_config=model_config,
            async_config=async_config
        )
        
        # Orchestrator
        self.orchestrator = IncrementalSentimentOrchestrator(
            content_loader=self.content_loader,
            sentiment_processor=self.sentiment_processor,
            sentiment_storage=self.sentiment_storage,
            sentiment_prescorer=self.prescorer,
            checkpoint_interval=self.config.processing_config.checkpoint_interval
        )
        
        # Deduplication
        deduplication_config = SentimentDeduplicationConfig(
            strategy=self.config.deduplication_config.strategy,
            conflict_resolution=self.config.deduplication_config.conflict_resolution,
            min_confidence_threshold=self.config.deduplication_config.min_confidence_threshold,
            min_intensity_threshold=self.config.deduplication_config.min_intensity_threshold,
            preserve_unique_tickers=self.config.deduplication_config.preserve_unique_tickers,
            primary_ticker_strategy=self.config.deduplication_config.primary_ticker_strategy,
            exclude_extraction_refusals=self.config.deduplication_config.exclude_extraction_refusals,
           prefer_explicit_tickers=self.config.deduplication_config.prefer_explicit_tickers
        )
        
        self.deduplicator = SentimentDeduplicator(config=deduplication_config)
        self.aggregator = TickerSentimentAggregator()
        
        # Storage
        self.storage = HybridConsensusStorage(
            local_path=Path(self.config.paths.consensus_data_path),
            s3_bucket=self.config.processing_config.s3_bucket,
            s3_prefix=self.config.paths.s3_prefix,
            local_retention_days=self.config.processing_config.local_retention_days
        )

        self.date_index = DateIndexStore(
            s3=self.storage.s3_storage.s3_client,      # already configured boto client
            bucket=self.config.processing_config.s3_bucket,
            key=f"{self.config.paths.s3_prefix}/metadata/date_index.json"
        )
    
    def _generate_date_range(self) -> List[str]:
        """Generate list of dates between start_date and end_date."""
        start = datetime.strptime(self.config.date_range.start_date, "%Y%m%d")
        end = datetime.strptime(self.config.date_range.end_date, "%Y%m%d")
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        
        return dates
    
    async def process_single_day(self, date: str) -> bool:
        """Process sentiment analysis for a single day."""
        try:
            logger.info(f"Processing date: {date}")
            
            # Process the day
            update = await self.orchestrator.process_day(date)
            self.orchestrator.finalize_day(date)
            
            # Load and deduplicate
            daily_batch = self.sentiment_storage.load_daily_batch(date)
            sub_deduplicated_sentiments = [
                self.deduplicator.deduplicate(x) 
                for x in daily_batch.submission_sentiments.values()
            ]
            # Aggregate and save
            daily_consensus = self.aggregator.aggregate_daily_sentiments(
                sub_deduplicated_sentiments, 
                date
            )
            self.storage.save_daily_consensus(daily_consensus)
            
            logger.info(f"Successfully processed date: {date}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing date {date}: {str(e)}")
            return False
    
    async def process_date_range(self):
        """Process sentiment analysis for the configured date range."""
        dates = self._generate_date_range()
        logger.info(f"Processing {len(dates)} dates from {dates[0]} to {dates[-1]}")
        
        successful = 0
        failed = 0
        done = self.date_index.processed_dates()
        logger.info(f"Already processed {len(done)} dates")
        for date in dates:
            if date in done:
                logger.info(f"Skipping {date} because it has already been processed")
                continue
            success = await self.process_single_day(date)
            if success:
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")
        return successful, failed

def load_config_from_yaml(config_path: str) -> SentimentAnalysisConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Convert string enum values back to actual enums
    if 'deduplication_config' in config_dict:
        dedup_config = config_dict['deduplication_config']
        
        # Convert strategy string to enum
        if 'strategy' in dedup_config:
            strategy_str = dedup_config['strategy']
            dedup_config['strategy'] = DeduplicationStrategy[strategy_str]
        
        # Convert conflict_resolution string to enum
        if 'conflict_resolution' in dedup_config:
            conflict_str = dedup_config['conflict_resolution']
            dedup_config['conflict_resolution'] = ConflictResolution[conflict_str]
    
    # Convert nested dictionaries to dataclass instances
    from dataclasses import fields
    
    def dict_to_dataclass(cls, data):
        field_types = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key in field_types:
                field_type = field_types[key]
                if hasattr(field_type, '__dataclass_fields__'):
                    kwargs[key] = dict_to_dataclass(field_type, value)
                else:
                    kwargs[key] = value
        return cls(**kwargs)
    
    return dict_to_dataclass(SentimentAnalysisConfig, config_dict)

async def main():
    """Main function to run the sentiment processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Reddit sentiment analysis for date range')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_yaml(args.config)
    
    # Initialize processor
    processor = SentimentProcessor(config)
    
    # Process date range
    await processor.process_date_range()

if __name__ == "__main__":
    asyncio.run(main())