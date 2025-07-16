from typing import Annotated, List, Literal, Optional, TypedDict, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

class TickerRelationship(BaseModel):
    ticker1: str = Field(description="First ticker in relationship")
    ticker2: Optional[str] = Field(description="Second ticker in relationship, if exists")
    relationship_type: Literal["inverse", "correlated", "causal"] = Field(
        description="Type of relationship between tickers"
    )
    description: str = Field(description="Brief description of the relationship")
    confidence: int = Field(
        description="Confidence score (1-10) for the relationship extraction. Do both tickers exist?",
        default=5
    )

class Sentiment(BaseModel):
    confidence: int = Field(
        description='Confidence score (1-10) for the extraction.',
        default=5
    )
    ticker: Optional[str] = Field(
        description='Ticker symbol of the stock being discussed.',
        default=None
    )
    is_implicit: bool = Field(
        description='Whether ticker was mentioned explicitly or inferred from context.',
        default=False
    )
    sentiment: Literal['bullish', 'bearish', 'neutral', 'unclear'] = Field(
        description='Overall sentiment direction.'
    )
    sentiment_intensity: int = Field(
        description='Intensity of the sentiment (1-10).',
        default=5
    )
    position: Literal['shares', 'calls', 'puts', 'spreads', 'watching', 'unclear'] = Field(
        description='Type of position discussed.',
        default='watching'
    )
    conviction: Literal['yolo', 'high', 'moderate', 'low', 'hedge', 'unclear'] = Field(
        description='Conviction level about the sentiment.',
        default='unclear'
    )
    time_horizon: Literal['day', 'swing', 'long', 'unclear'] = Field(
        description='Time horizon for the position.',
        default='unclear'
    )
    source: Literal['submission', 'comments', 'both'] = Field(
        description='Where this sentiment was extracted from.',
        default='both'
    )
    explanation: str = Field(
        description='Explanation of the sentiment extraction.',
        default=''
    )
    extraction_refusal: bool = Field(
        description='Whether the sentiment extraction was refused.',
        default=False
    )

class SentimentResults(BaseModel):
    primary_ticker: Optional[str] = Field(
        description='Primary ticker discussed in the submission, if identifiable.',
        default=None
    )
    sentiments: List[Sentiment] = Field(
        description='List of extracted sentiments from the discussion.',
        default_factory=list
    )
    relationships: Optional[List[TickerRelationship]] = Field(
        description='Relationships between different tickers mentioned.',
        default_factory=list
    )

class ExtractionError(Exception):
    """Custom exception for extraction failures"""
    pass

@dataclass
class RedditContent:
    """Clean data structure for Reddit content"""
    title: str
    submission_body: str
    comments: List[str]
    submission_id: Optional[str] = None
    created_utc: Optional[datetime] = None
    subreddit: Optional[str] = None

@dataclass
class ModelConfig:
    """Configuration for the LLM model"""
    model_name: str = field(default="gpt-4.1-nano")
    temperature: float = field(default=0.0, metadata={"ge": 0.0, "le": 2.0})
    max_tokens: Optional[int] = field(default=None)
    timeout: Optional[int] = field(default=30)
    max_retries: int = field(default=2)

    def __post_init__(self):
        """Validate field constraints after initialization"""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

@dataclass
class AsyncExtractionConfig:
    """Configuration for async extraction"""
    max_concurrent_requests: int = 50  # Max concurrent requests
    requests_per_minute: int = 500     # API rate limit (RPM)
    tokens_per_minute: int = 200000    # API token limit (TPM)
    batch_size: int = 100              # Process in batches of this size
    delay_between_batches: float = 1.0 # Delay between batches (seconds)
    enable_rate_limiting: bool = True  # Enable automatic rate limiting
    token_buffer_ratio: float = 0.9    # Use 90% of token limit as safety buffer
    request_buffer_ratio: float = 0.9  # Use 90% of request limit as safety buffer

@dataclass
class ProcessingResult:
    """Result from processing a single submission"""
    submission_id: str
    sentiments: List[SentimentResults]
    processing_time: float
    extraction_count: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class BatchProcessingResult:
    """Result from processing a batch of submissions"""
    results: Dict[str, ProcessingResult]  # submission_id -> ProcessingResult
    total_submissions: int
    successful_submissions: int
    failed_submissions: int
    total_extractions: int
    processing_time: float
    timestamp: datetime

@dataclass
class DailyUpdate:
    """Update for a specific day - can be partial or complete"""
    date: str
    submission_results: Dict[str, ProcessingResult]
    is_complete_day: bool  # True if this represents all submissions for the day
    update_timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class DailySentimentBatch:
    """Complete sentiment data for a single day"""
    date: str
    submission_sentiments: Dict[str, List[SentimentResults]]  # submission_id -> sentiments
    processing_metadata: Dict[str, Any]
    last_updated: datetime
    total_submissions: int
    total_extractions: int

@dataclass
class ProcessingCheckpoint:
    """
    Checkpoint for tracking processing progress for a specific date.
    
    This stores the current state of processing for a day, allowing us to
    resume processing exactly where we left off if interrupted.
    """
    date: str                           # Date being processed (YYYYMMDD)
    processed_submissions: Set[str]     # submission_ids that have been successfully processed
    failed_submissions: Set[str]        # submission_ids that failed processing
    last_checkpoint_time: datetime      # When this checkpoint was last updated
    processing_session_id: str          # Unique ID for the current processing session
    total_submissions_found: int        # Total number of submissions found for this date
    metadata: Dict[str, Any] = None     # Additional metadata (processing times, session info, etc.)
    
    def __post_init__(self):
        """Initialize metadata if None"""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def processed_count(self) -> int:
        """Number of successfully processed submissions"""
        return len(self.processed_submissions)
    
    @property
    def failed_count(self) -> int:
        """Number of failed submissions"""
        return len(self.failed_submissions)
    
    @property
    def total_handled(self) -> int:
        """Total submissions handled (processed + failed)"""
        return len(self.processed_submissions) + len(self.failed_submissions)
    
    @property
    def remaining_count(self) -> int:
        """Number of submissions still to be processed"""
        return max(0, self.total_submissions_found - self.total_handled)
    
    @property
    def completion_rate(self) -> float:
        """Completion rate as a fraction (0.0 to 1.0)"""
        if self.total_submissions_found == 0:
            return 1.0
        return self.total_handled / self.total_submissions_found
    
    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)"""
        if self.total_handled == 0:
            return 0.0
        return len(self.processed_submissions) / self.total_handled
    
    @property
    def is_complete(self) -> bool:
        """Whether processing is complete for this date"""
        return self.remaining_count == 0
    
    def add_processed(self, submission_id: str):
        """Mark a submission as successfully processed"""
        self.processed_submissions.add(submission_id)
        # Remove from failed if it was there (in case of retry)
        self.failed_submissions.discard(submission_id)
        self.last_checkpoint_time = datetime.now()
    
    def add_failed(self, submission_id: str):
        """Mark a submission as failed"""
        self.failed_submissions.add(submission_id)
        # Remove from processed if it was there (shouldn't happen, but safety)
        self.processed_submissions.discard(submission_id)
        self.last_checkpoint_time = datetime.now()
    
    def reset_failed(self, submission_ids: Set[str] = None):
        """
        Reset failed submissions for retry.
        
        Args:
            submission_ids: If provided, only reset these specific submissions.
                          If None, reset all failed submissions.
        """
        if submission_ids is None:
            self.failed_submissions.clear()
        else:
            self.failed_submissions -= submission_ids
        self.last_checkpoint_time = datetime.now()
    
    def update_total_found(self, new_total: int):
        """Update total submissions found (in case new submissions are discovered)"""
        if new_total > self.total_submissions_found:
            self.total_submissions_found = new_total
            self.last_checkpoint_time = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the checkpoint status"""
        return {
            'date': self.date,
            'processed': self.processed_count,
            'failed': self.failed_count,
            'total_found': self.total_submissions_found,
            'remaining': self.remaining_count,
            'completion_rate': f"{self.completion_rate:.1%}",
            'success_rate': f"{self.success_rate:.1%}",
            'is_complete': self.is_complete,
            'last_updated': self.last_checkpoint_time.isoformat(),
            'session_id': self.processing_session_id
        }

    @property
    def is_processing_complete(self) -> bool:
        """Whether all submissions have been processed (successfully or failed)"""
        return self.total_handled >= self.total_submissions_found
    
    @property
    def can_be_finalized(self) -> bool:
        """Whether this day can be finalized (all processing complete)"""
        return self.is_processing_complete and not self.is_finalized

@dataclass
class IncrementalDailyUpdate:
    """Update that can be applied incrementally"""
    date: str
    new_submission_results: Dict[str, ProcessingResult]  # Only new results
    checkpoint: ProcessingCheckpoint
    update_timestamp: datetime
    is_final: bool = False  # True when all submissions for day are processed


class DeduplicationStrategy(Enum):
    """Strategy for resolving conflicting sentiments"""
    HIGHEST_CONFIDENCE = "highest_confidence"      # Take sentiment with highest confidence
    CONFIDENCE_WEIGHTED = "confidence_weighted"    # Weight by confidence scores
    MAJORITY_VOTE = "majority_vote"               # Most common sentiment wins
    INTENSITY_WEIGHTED = "intensity_weighted"     # Weight by sentiment_intensity scores
    CONVICTION_WEIGHTED = "conviction_weighted"   # Weight by conviction level

class ConflictResolution(Enum):
    """How to handle conflicting sentiments for same ticker"""
    MERGE = "merge"           # Combine into single sentiment
    KEEP_STRONGEST = "keep_strongest"  # Keep only the strongest sentiment
    DROP_LOW_CONFIDENCE = "drop_low_confidence"  # Remove low confidence conflicts

@dataclass
class DeduplicationConfig:
    """Configuration for deduplication behavior"""
    strategy: DeduplicationStrategy = DeduplicationStrategy.CONFIDENCE_WEIGHTED
    conflict_resolution: ConflictResolution = ConflictResolution.MERGE
    min_confidence_threshold: int = 5
    min_intensity_threshold: int = 3
    preserve_unique_tickers: bool = True
    primary_ticker_strategy: str = "most_frequent"  # or "highest_confidence"
    exclude_extraction_refusals: bool = True
    prefer_explicit_tickers: bool = True  # Prefer explicit over implicit tickers

@dataclass
class ConflictInfo:
    """Information about a sentiment conflict"""
    ticker: str
    conflicting_sentiments: List['Sentiment']
    resolution_method: str
    final_sentiment: 'Sentiment'
    confidence_spread: float  # Max confidence - Min confidence
    conviction_levels: List[str]  # Different conviction levels involved


@dataclass
class TickerAggregateStats:
    """Aggregated statistics for a single ticker across all submissions"""
    ticker: str
    total_mentions: int
    sentiment_distribution: Dict[str, int]  # {'bullish': 5, 'bearish': 2, 'neutral': 3}
    
    # Weighted averages
    avg_confidence: float
    avg_sentiment_intensity: float
    confidence_weighted_intensity: float
    
    # Categorical distributions
    position_distribution: Dict[str, int]
    conviction_distribution: Dict[str, int] 
    time_horizon_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    
    # Submission-level stats
    unique_submissions: int
    submissions_as_primary: int  # How many times this was the primary ticker
    
    # Quality metrics
    explicit_mentions: int
    implicit_mentions: int
    avg_confidence_by_sentiment: Dict[str, float]
    
    # Consensus metrics
    sentiment_consensus_strength: float  # 0-1, how unified the sentiment is
    dominant_sentiment: str
    dominant_sentiment_percentage: float
    
    # NEW: Polarization and conviction metrics
    sentiment_polarization: float = 0.0  # 0-1, higher = more polarized
    sentiment_controversy_score: float = 0.0  # Custom controversy metric
    is_controversial: bool = False  # Flag for highly polarized stocks
    conviction_weighted_bullish_score: float = 0.0
    conviction_weighted_bearish_score: float = 0.0
    avg_bullish_conviction: float = 0.0
    avg_bearish_conviction: float = 0.0
    date: str = None

@dataclass 
class DailyTickerConsensus:
    """Daily aggregated ticker sentiment consensus"""
    date: str
    ticker_stats: Dict[str, TickerAggregateStats]
    total_submissions_processed: int
    total_sentiments_processed: int
    unique_tickers: int
    
    # Methods for cross-ticker insights
    def get_most_mentioned(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get tickers sorted by total mentions"""
        return sorted(
            [(ticker, stats.total_mentions) for ticker, stats in self.ticker_stats.items()],
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
    
    def get_conviction_weighted_bullish(self, limit: int = 10, min_mentions: int = 2) -> List[Tuple[str, float, float, int, int, float]]:
        """Get bullish tickers weighted by conviction level"""
        
        weighted_bullish = []
        for ticker, stats in self.ticker_stats.items():
            if stats.total_mentions < min_mentions:
                continue
            
            bullish_count = stats.sentiment_distribution.get('bullish', 0)
            if bullish_count == 0:
                continue
                
            bullish_pct = (bullish_count / stats.total_mentions) * 100
            
            weighted_bullish.append((
                ticker,
                stats.conviction_weighted_bullish_score,  # Pre-calculated
                bullish_pct,
                bullish_count,
                stats.total_mentions,
                stats.avg_bullish_conviction
            ))
        
        return sorted(weighted_bullish, key=lambda x: (x[1], x[2]), reverse=True)[:limit]
    
    def get_conviction_weighted_bearish(self, limit: int = 10, min_mentions: int = 2) -> List[Tuple[str, float, float, int, int, float]]:
        """Get bearish tickers weighted by conviction level"""
        
        weighted_bearish = []
        for ticker, stats in self.ticker_stats.items():
            if stats.total_mentions < min_mentions:
                continue
            
            bearish_count = stats.sentiment_distribution.get('bearish', 0)
            if bearish_count == 0:
                continue
                
            bearish_pct = (bearish_count / stats.total_mentions) * 100
            
            weighted_bearish.append((
                ticker,
                stats.conviction_weighted_bearish_score,  # Pre-calculated
                bearish_pct,
                bearish_count,
                stats.total_mentions,
                stats.avg_bearish_conviction
            ))
        
        return sorted(weighted_bearish, key=lambda x: (x[1], x[2]), reverse=True)[:limit]
    
    def get_most_bullish(self, limit: int = 10, min_mentions: int = 2, 
                        exclude_controversial: bool = False) -> List[Tuple[str, float, int, int, float]]:
        """Get most bullish tickers with option to exclude controversial ones"""
        
        bullish_tickers = []
        for ticker, stats in self.ticker_stats.items():
            if stats.total_mentions < min_mentions:
                continue
                
            # Skip controversial stocks if requested
            if exclude_controversial and stats.is_controversial:
                continue
                
            bullish_count = stats.sentiment_distribution.get('bullish', 0)
            bullish_pct = (bullish_count / stats.total_mentions) * 100
            
            bullish_tickers.append((
                ticker, 
                bullish_pct, 
                bullish_count, 
                stats.total_mentions,
                stats.sentiment_controversy_score
            ))
        
        return sorted(bullish_tickers, key=lambda x: (x[1], x[2]), reverse=True)[:limit]
    
    def get_most_bearish(self, limit: int = 10, min_mentions: int = 2, 
                        exclude_controversial: bool = False) -> List[Tuple[str, float, int, int, float]]:
        """Get most bearish tickers with option to exclude controversial ones"""
        
        bearish_tickers = []
        for ticker, stats in self.ticker_stats.items():
            if stats.total_mentions < min_mentions:
                continue
                
            # Skip controversial stocks if requested
            if exclude_controversial and stats.is_controversial:
                continue
                
            bearish_count = stats.sentiment_distribution.get('bearish', 0)
            bearish_pct = (bearish_count / stats.total_mentions) * 100
            
            bearish_tickers.append((
                ticker, 
                bearish_pct, 
                bearish_count, 
                stats.total_mentions,
                stats.sentiment_controversy_score
            ))
        
        return sorted(bearish_tickers, key=lambda x: (x[1], x[2]), reverse=True)[:limit]

    def get_most_controversial(self, limit: int = 10, min_mentions: int = 3) -> List[Tuple[str, float, float, float, int]]:
        """Get most controversial/polarized tickers"""
        
        controversial_tickers = []
        for ticker, stats in self.ticker_stats.items():
            if stats.total_mentions < min_mentions:
                continue
                
            bullish_pct = stats.sentiment_distribution.get('bullish', 0) / stats.total_mentions * 100
            bearish_pct = stats.sentiment_distribution.get('bearish', 0) / stats.total_mentions * 100
            
            controversial_tickers.append((
                ticker,
                stats.sentiment_controversy_score,
                bullish_pct,
                bearish_pct,
                stats.total_mentions
            ))
        
        return sorted(controversial_tickers, key=lambda x: x[1], reverse=True)[:limit]

    def get_clear_sentiment_bullish(self, limit: int = 10, min_mentions: int = 2) -> List[Tuple[str, float, int, int, float]]:
        """Get clearly bullish tickers (high bullish %, low bearish %)"""
        
        clear_bullish = []
        for ticker, stats in self.ticker_stats.items():
            if stats.total_mentions < min_mentions:
                continue
                
            bullish_count = stats.sentiment_distribution.get('bullish', 0)
            bearish_count = stats.sentiment_distribution.get('bearish', 0)
            
            bullish_pct = (bullish_count / stats.total_mentions) * 100
            bearish_pct = (bearish_count / stats.total_mentions) * 100
            
            # Only include if bullish > 60% AND bearish < 20%
            if bullish_pct > 60 and bearish_pct < 20:
                clear_bullish.append((
                    ticker,
                    bullish_pct,
                    bullish_count,
                    stats.total_mentions,
                    bearish_pct  # Include for context
                ))
        
        return sorted(clear_bullish, key=lambda x: x[1], reverse=True)[:limit]
    
    def get_clear_sentiment_bearish(self, limit: int = 10, min_mentions: int = 2) -> List[Tuple[str, float, int, int, float]]:
        """Get clearly bearish tickers (high bearish %, low bullish %)"""
        
        clear_bearish = []
        for ticker, stats in self.ticker_stats.items():
            if stats.total_mentions < min_mentions:
                continue
                
            bullish_count = stats.sentiment_distribution.get('bullish', 0)
            bearish_count = stats.sentiment_distribution.get('bearish', 0)
            
            bullish_pct = (bullish_count / stats.total_mentions) * 100
            bearish_pct = (bearish_count / stats.total_mentions) * 100
            
            # Only include if bearish > 60% AND bullish < 20%
            if bearish_pct > 60 and bullish_pct < 20:
                clear_bearish.append((
                    ticker,
                    bearish_pct,
                    bearish_count,
                    stats.total_mentions,
                    bullish_pct  # Include for context
                ))
        
        return sorted(clear_bearish, key=lambda x: x[1], reverse=True)[:limit]
    
    def get_highest_conviction(self, limit: int = 10, min_mentions: int = 2) -> List[Tuple[str, float, int]]:
        """Get tickers with highest conviction levels"""
        conviction_weights = {'yolo': 10, 'high': 8, 'moderate': 6, 'low': 4, 'hedge': 3, 'unclear': 1}
        
        conviction_tickers = []
        for ticker, stats in self.ticker_stats.items():
            if stats.total_mentions >= min_mentions:
                weighted_conviction = sum(
                    count * conviction_weights.get(conviction, 1) 
                    for conviction, count in stats.conviction_distribution.items()
                ) / stats.total_mentions
                conviction_tickers.append((ticker, weighted_conviction, stats.total_mentions))
        
        return sorted(conviction_tickers, key=lambda x: x[1], reverse=True)[:limit]
    
    def get_ticker_summary(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary for a specific ticker"""
        if ticker not in self.ticker_stats:
            return None
        
        stats = self.ticker_stats[ticker]
        return {
            'ticker': ticker,
            'total_mentions': stats.total_mentions,
            'unique_submissions': stats.unique_submissions,
            'primary_ticker_count': stats.submissions_as_primary,
            'sentiment_breakdown': stats.sentiment_distribution,
            'dominant_sentiment': f"{stats.dominant_sentiment} ({stats.dominant_sentiment_percentage}%)",
            'consensus_strength': stats.sentiment_consensus_strength,
            'avg_confidence': stats.avg_confidence,
            'conviction_breakdown': stats.conviction_distribution,
            'position_breakdown': stats.position_distribution,
            'explicit_vs_implicit': f"{stats.explicit_mentions}/{stats.implicit_mentions}",
            'is_controversial': stats.is_controversial,
            'controversy_score': stats.sentiment_controversy_score,
            'polarization': stats.sentiment_polarization
        }
    
    def get_daily_summary(self) -> Dict[str, Any]:
        """Get high-level daily summary"""
        return {
            'date': self.date,
            'total_submissions': self.total_submissions_processed,
            'total_sentiments': self.total_sentiments_processed,
            'unique_tickers': self.unique_tickers,
            'top_5_mentioned': self.get_most_mentioned(5),
            'top_5_bullish': self.get_most_bullish(5),
            'top_5_bearish': self.get_most_bearish(5),
            'highest_conviction': self.get_highest_conviction(5)
        }
    
    def get_sentiment_summary(self) -> Dict[str, List[tuple]]:
        """Get comprehensive sentiment breakdown"""
        
        return {
            'clear_bullish': self.get_clear_sentiment_bullish(10),
            'clear_bearish': self.get_clear_sentiment_bearish(10), 
            'most_controversial': self.get_most_controversial(10),
            'highest_conviction_bullish': self.get_conviction_weighted_bullish(10),
            'most_mentioned': self.get_most_mentioned(10)
        }
    
    # You can also use @property for computed properties
    @property
    def most_active_ticker(self) -> Optional[str]:
        """Ticker with most mentions"""
        if not self.ticker_stats:
            return None
        return max(self.ticker_stats.items(), key=lambda x: x[1].total_mentions)[0]
    
    @property
    def sentiment_overview(self) -> Dict[str, int]:
        """Overall sentiment distribution across all tickers"""
        total_sentiments = defaultdict(int)
        for stats in self.ticker_stats.values():
            for sentiment, count in stats.sentiment_distribution.items():
                total_sentiments[sentiment] += count
        return dict(total_sentiments)

@dataclass
class PathConfig:
    reddit_comments_path: str = "../reddit_comments"
    processed_sentiments_path: str = "../data/processed_sentiments"
    relevance_model_path: str = "../data/models/relevance_classifier_v1"
    consensus_data_path: str = "../data/consensus_data"
    s3_prefix: str = "consensus_data"

@dataclass
class BatchPathConfig:
    reddit_comments_path:      str = "../reddit_comments"
    processed_sentiments_path: str = "../data/processed_sentiments"
    relevance_model_path:      str = "../data/models/relevance_classifier_v1"
    consensus_data_path:       str = "../data/consensus_data"
    s3_prefix:                 str = "consensus_data"
    batch_jobs_path:    str = "../batch_jobs"      # where JSONL + metadata live
    batch_outputs_path: str = "../batch_outputs"   # downloaded result files

@dataclass
class ProcessingConfig:
    max_n_comments: int = 20
    relevance_threshold: float = 0.5
    checkpoint_interval: int = 3
    s3_bucket: Optional[str] = "internetwisdom-data"
    local_retention_days: int = 30

@dataclass
class DateRangeConfig:
    start_date: str  # Format: YYYYMMDD
    end_date: str    # Format: YYYYMMDD

@dataclass
class SentimentAnalysisConfig:
    paths: PathConfig
    async_config: AsyncExtractionConfig
    model_config: ModelConfig
    deduplication_config: DeduplicationConfig
    processing_config: ProcessingConfig
    date_range: DateRangeConfig


# ---------------------------- CONFIG ADD-ON ---------------------------
@dataclass
class BatchConfig:
    max_active_jobs: int = 10
    poll_seconds: int = 120            # how often to poll
    submit_per_day: bool = True        # one job per calendar day?


@dataclass
class FlagsConfig:
    force_overwrite: bool = False

@dataclass
class SentimentAnalysisConfigWithBatch:
    paths:                 BatchPathConfig
    async_config:          AsyncExtractionConfig
    model_config:          ModelConfig
    deduplication_config:  DeduplicationConfig
    processing_config:     ProcessingConfig
    date_range:            DateRangeConfig
    batch:                 BatchConfig  
    flags:                 FlagsConfig

