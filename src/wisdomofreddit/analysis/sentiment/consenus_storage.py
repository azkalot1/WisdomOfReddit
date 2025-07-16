from typing import Protocol, Optional, Iterable, List, Dict, Any, Set
from pathlib import Path
import json
import pickle
import boto3
import io
from datetime import datetime, date, timedelta
import logging
from dataclasses import asdict
import json
from .aggregation import DailyTickerConsensus, TickerAggregateStats

logger = logging.getLogger(__name__)

class ConsensusStorage(Protocol):
    """Interface for storing ticker consensus data"""
    
    def save_daily_consensus(self, consensus: DailyTickerConsensus) -> None:
        """Save daily ticker consensus"""
        ...
    
    def load_daily_consensus(self, date: str) -> Optional[DailyTickerConsensus]:
        """Load daily consensus for a specific date"""
        ...
    
    def get_ticker_history(self, ticker: str, start_date: str, end_date: str) -> List[TickerAggregateStats]:
        """Get historical data for a specific ticker"""
        ...
    
    def get_date_range_available(self) -> tuple[str, str]:
        """Get earliest and latest available dates"""
        ...
    
    def list_available_dates(self) -> List[str]:
        """List all dates with consensus data"""
        ...

class LocalConsensusStorage:
    """Local file-based consensus storage optimized for different access patterns"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Organized structure for different access patterns
        self.daily_consensus_path = self.base_path / "daily_consensus"
        self.metadata_path = self.base_path / "metadata"
        
        # FIX: Remove ticker_history_path reference
        for path in [self.daily_consensus_path, self.metadata_path]:
            path.mkdir(exist_ok=True)
        
        self.date_index_path = self.metadata_path / "date_index.json"
        self.ticker_index_path = self.metadata_path / "ticker_index.json"
        
        self._load_indexes()
    
    def _load_indexes(self):
        """Load indexes for fast lookups"""
        # Date index: {date: {ticker_count, file_size, last_updated}}
        if self.date_index_path.exists():
            with open(self.date_index_path, 'r') as f:
                self.date_index = json.load(f)
        else:
            self.date_index = {}
        
        # Ticker index: {ticker: [dates_available]}
        if self.ticker_index_path.exists():
            with open(self.ticker_index_path, 'r') as f:
                self.ticker_index = json.load(f)
        else:
            self.ticker_index = {}
    
    def _save_indexes(self):
        """Save indexes atomically"""
        temp_date_path = self.date_index_path.with_suffix('.tmp')
        temp_ticker_path = self.ticker_index_path.with_suffix('.tmp')
        
        try:
            with open(temp_date_path, 'w') as f:
                json.dump(self.date_index, f, indent=2, default=str)
            with open(temp_ticker_path, 'w') as f:
                json.dump(self.ticker_index, f, indent=2, default=str)
            
            temp_date_path.replace(self.date_index_path)
            temp_ticker_path.replace(self.ticker_index_path)
            
        except Exception as e:
            for temp_path in [temp_date_path, temp_ticker_path]:
                if temp_path.exists():
                    temp_path.unlink()
            raise e
    
    def save_daily_consensus(self, consensus: DailyTickerConsensus) -> None:
        """Save daily consensus"""
        
        # Save complete daily consensus (for date-based queries)
        daily_file = self.daily_consensus_path / f"{consensus.date}_consensus.pkl"
        with open(daily_file, 'wb') as f:
            pickle.dump(consensus, f)
        
        # Update indexes
        self.date_index[consensus.date] = {
            'ticker_count': consensus.unique_tickers,
            'total_submissions': consensus.total_submissions_processed,
            'total_sentiments': consensus.total_sentiments_processed,
            'file_size': daily_file.stat().st_size,
            'tickers': list(consensus.ticker_stats.keys()),  # Add for ticker lookups
            'last_updated': datetime.now().isoformat()
        }
        
        for ticker in consensus.ticker_stats.keys():
            if ticker not in self.ticker_index:
                self.ticker_index[ticker] = []
            if consensus.date not in self.ticker_index[ticker]:
                self.ticker_index[ticker].append(consensus.date)
                self.ticker_index[ticker].sort()
        
        self._save_indexes()
        
        logger.info(f"✅ Saved consensus for {consensus.date}: "
                   f"{consensus.unique_tickers} tickers, "
                   f"{consensus.total_submissions_processed} submissions")
    
    def load_daily_consensus(self, date: str) -> Optional[DailyTickerConsensus]:
        """Load complete daily consensus"""
        daily_file = self.daily_consensus_path / f"{date}_consensus.pkl"
        
        if not daily_file.exists():
            return None
        
        try:
            with open(daily_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load consensus for {date}: {e}")
            return None
    
    # FIX: Add missing method
    def get_ticker_history(self, ticker: str, start_date: str, end_date: str) -> List[TickerAggregateStats]:
        """Get historical data for a specific ticker"""
        if ticker not in self.ticker_index:
            return []
        
        # Filter dates in range
        available_dates = [
            d for d in self.ticker_index[ticker] 
            if start_date <= d <= end_date
        ]
        
        ticker_history = []
        for date in available_dates:
            consensus = self.load_daily_consensus(date)
            if consensus and ticker in consensus.ticker_stats:
                ts = consensus.ticker_stats[ticker]
                ts.date = date
                ticker_history.append(ts)
        
        return ticker_history
    
    def get_date_range_available(self) -> tuple[str, str]:
        """Get earliest and latest available dates"""
        if not self.date_index:
            return None, None
        
        dates = sorted(self.date_index.keys())
        return dates[0], dates[-1]
    
    def list_available_dates(self) -> List[str]:
        """List all dates with consensus data"""
        return sorted(self.date_index.keys())
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = sum(
            (self.daily_consensus_path / f"{date}_consensus.pkl").stat().st_size
            for date in self.date_index.keys()
            if (self.daily_consensus_path / f"{date}_consensus.pkl").exists()
        )
        
        return {
            'total_dates': len(self.date_index),
            'total_tickers_tracked': len(self.ticker_index),
            'date_range': self.get_date_range_available(),
            'total_storage_mb': total_size / (1024 * 1024),
            'avg_tickers_per_day': sum(
                info['ticker_count'] for info in self.date_index.values()
            ) / len(self.date_index) if self.date_index else 0
        }

class S3ConsensusStorage:
    """S3-based consensus storage for scalability"""
    
    def __init__(self, bucket_name: str, prefix: str = "consensus_data"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3')
        
        # Local cache for indexes
        self.cache_path = Path.home() / ".reddit_sentiment_cache"
        self.cache_path.mkdir(exist_ok=True)
        
        self._sync_indexes()
    
    def _get_s3_key(self, path: str) -> str:
        """Generate S3 key with prefix"""
        return f"{self.prefix}/{path}"
    
    def _sync_indexes(self):
        """Sync indexes from S3 to local cache"""
        try:
            # Download date index
            date_index_key = self._get_s3_key("metadata/date_index.json")
            local_date_index = self.cache_path / "date_index.json"
            
            self.s3_client.download_file(
                self.bucket_name, date_index_key, str(local_date_index)
            )
            
            with open(local_date_index, 'r') as f:
                self.date_index = json.load(f)
            
            # Download ticker index
            ticker_index_key = self._get_s3_key("metadata/ticker_index.json")
            local_ticker_index = self.cache_path / "ticker_index.json"
            
            self.s3_client.download_file(
                self.bucket_name, ticker_index_key, str(local_ticker_index)
            )
            
            with open(local_ticker_index, 'r') as f:
                self.ticker_index = json.load(f)
                
            logger.info(f"Synced indexes: {len(self.date_index)} dates, {len(self.ticker_index)} tickers")
            
        except Exception as e:
            logger.warning(f"Could not sync indexes from S3: {e}")
            self.date_index = {}
            self.ticker_index = {}
    
    def save_daily_consensus(self, consensus: DailyTickerConsensus) -> None:
        """Save to S3 with local caching"""
        
        # Save complete daily consensus
        daily_key = self._get_s3_key(f"daily_consensus/{consensus.date}_consensus.pkl")
        
        # Serialize to bytes
        buffer = io.BytesIO()
        pickle.dump(consensus, buffer)
        buffer.seek(0)
        
        self.s3_client.upload_fileobj(buffer, self.bucket_name, daily_key)
        
        # Update and upload indexes
        self.date_index[consensus.date] = {
            'ticker_count': consensus.unique_tickers,
            'total_submissions': consensus.total_submissions_processed,
            'total_sentiments': consensus.total_sentiments_processed,
            'tickers': list(consensus.ticker_stats.keys()),
            'last_updated': datetime.now().isoformat(),
            's3_key': daily_key
        }
        
        for ticker in consensus.ticker_stats.keys():
            if ticker not in self.ticker_index:
                self.ticker_index[ticker] = []
            if consensus.date not in self.ticker_index[ticker]:
                self.ticker_index[ticker].append(consensus.date)
                self.ticker_index[ticker].sort()
        
        self._upload_indexes()
        
        logger.info(f"✅ Uploaded consensus to S3 for {consensus.date}")
    
    def _upload_indexes(self):
        """Upload updated indexes to S3"""
        # Upload date index
        date_index_buffer = io.BytesIO()
        date_index_buffer.write(json.dumps(self.date_index, indent=2, default=str).encode())
        date_index_buffer.seek(0)
        
        self.s3_client.upload_fileobj(
            date_index_buffer, 
            self.bucket_name, 
            self._get_s3_key("metadata/date_index.json")
        )
        
        # Upload ticker index
        ticker_index_buffer = io.BytesIO()
        ticker_index_buffer.write(json.dumps(self.ticker_index, indent=2, default=str).encode())
        ticker_index_buffer.seek(0)
        
        self.s3_client.upload_fileobj(
            ticker_index_buffer,
            self.bucket_name,
            self._get_s3_key("metadata/ticker_index.json")
        )
    
    def load_daily_consensus(self, date: str) -> Optional[DailyTickerConsensus]:
        """Load from S3 with local caching"""
        if date not in self.date_index:
            return None
        
        # Check local cache first
        cache_file = self.cache_path / f"{date}_consensus.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read failed for {date}: {e}")
        
        # Download from S3
        try:
            daily_key = self._get_s3_key(f"daily_consensus/{date}_consensus.pkl")
            
            # Download to cache
            self.s3_client.download_file(self.bucket_name, daily_key, str(cache_file))
            
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load consensus from S3 for {date}: {e}")
            return None
    
    # FIX: Add missing Protocol methods
    def get_ticker_history(self, ticker: str, start_date: str, end_date: str) -> List[TickerAggregateStats]:
        """Get historical data for a specific ticker"""
        if ticker not in self.ticker_index:
            return []
        
        available_dates = [
            d for d in self.ticker_index[ticker] 
            if start_date <= d <= end_date
        ]
        
        ticker_history = []
        for date in available_dates:
            consensus = self.load_daily_consensus(date)
            if consensus and ticker in consensus.ticker_stats:
                ts = consensus.ticker_stats[ticker]
                ts.date = date
                ticker_history.append(ts)
        
        return ticker_history
    
    def get_date_range_available(self) -> tuple[str, str]:
        """Get earliest and latest available dates"""
        if not self.date_index:
            return None, None
        
        dates = sorted(self.date_index.keys())
        return dates[0], dates[-1]
    
    def list_available_dates(self) -> List[str]:
        """List all dates with consensus data"""
        return sorted(self.date_index.keys())

class HybridConsensusStorage:
    """Hybrid storage: local for recent data, S3 for historical"""
    
    def __init__(self, local_path: Path, s3_bucket: str, s3_prefix: str,
                 local_retention_days: int = 30):
        self.local_storage = LocalConsensusStorage(local_path)
        self.s3_storage = S3ConsensusStorage(s3_bucket, s3_prefix)
        self.local_retention_days = local_retention_days
    
    def save_daily_consensus(self, consensus: DailyTickerConsensus) -> None:
        """Save to both local and S3"""
        # Always save locally first (faster)
        self.local_storage.save_daily_consensus(consensus)
        
        # Also save to S3 for backup/historical access
        try:
            self.s3_storage.save_daily_consensus(consensus)
        except Exception as e:
            logger.error(f"Failed to backup to S3: {e}")
    
    def load_daily_consensus(self, date: str) -> Optional[DailyTickerConsensus]:
        """Load from local first, fallback to S3"""
        # Try local first
        result = self.local_storage.load_daily_consensus(date)
        if result:
            return result
        
        # Fallback to S3
        logger.info(f"Loading {date} from S3 (not in local storage)")
        return self.s3_storage.load_daily_consensus(date)
    
    # FIX: Add missing Protocol methods
    def get_ticker_history(self, ticker: str, start_date: str, end_date: str) -> List[TickerAggregateStats]:
        """Get ticker history from both sources"""
        # Try local first, then S3 for missing dates
        local_history = self.local_storage.get_ticker_history(ticker, start_date, end_date)
        s3_history = self.s3_storage.get_ticker_history(ticker, start_date, end_date)
        
        # Merge and deduplicate by date
        combined = {}
        for stats in local_history + s3_history:
            # Assuming TickerAggregateStats has a date field or we can derive it
            combined[stats.ticker] = stats  # This might need adjustment based on your data structure
        
        return list(combined.values())
    
    def get_date_range_available(self) -> tuple[str, str]:
        """Get combined date range from both sources"""
        local_range = self.local_storage.get_date_range_available()
        s3_range = self.s3_storage.get_date_range_available()
        
        # Combine ranges
        start_dates = [r[0] for r in [local_range, s3_range] if r[0] is not None]
        end_dates = [r[1] for r in [local_range, s3_range] if r[1] is not None]
        
        if not start_dates:
            return None, None
        
        return min(start_dates), max(end_dates)
    
    def list_available_dates(self) -> List[str]:
        """List all dates from both sources"""
        local_dates = set(self.local_storage.list_available_dates())
        s3_dates = set(self.s3_storage.list_available_dates())
        return sorted(local_dates.union(s3_dates))
    
    def cleanup_old_local_data(self):
        """Remove old local data based on retention policy"""
        cutoff_date = (datetime.now() - timedelta(days=self.local_retention_days)).strftime('%Y%m%d')
        
        old_dates = [
            date for date in self.local_storage.list_available_dates()
            if date < cutoff_date
        ]
        
        for date in old_dates:
            # Remove local files but keep in S3
            daily_file = self.local_storage.daily_consensus_path / f"{date}_consensus.pkl"
            if daily_file.exists():
                daily_file.unlink()
                logger.debug(f"Removed old local file: {date}")
        
        logger.info(f"Cleaned up {len(old_dates)} old local consensus files")


class DateIndexStore:
    """
    Very small wrapper around the JSON date-index file:
        {
          "20200101": {...optional metadata...},
          "20200102": {...}
        }
    """

    def __init__(self, *, s3, bucket: str, key: str):
        """
        Parameters
        ----------
        s3     : boto3-like object that has .get_object / .put_object
        bucket : 'internetwisdom-data'
        key    : 'consensus_data/metadata/date_index.json'
        """
        self.s3     = s3
        self.bucket = bucket
        self.key    = key
        self._cache: Set[str] | None = None   # lazy load

    # ---------- public API -------------------------------------------
    def processed_dates(self) -> Set[str]:
        if self._cache is None:
            self._cache = self._load_from_s3()
        return self._cache

    def mark_processed(self, dates: Iterable[str]):
        dates = set(dates)
        if not dates:
            return
        idx = self.processed_dates() | dates
        self._save_to_s3(idx)
        self._cache = idx                        # update cache

    # ---------- private helpers --------------------------------------
    def _load_from_s3(self) -> Set[str]:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            data = json.loads(obj["Body"].read())
            return set(data.keys())
        except self.s3.exceptions.NoSuchKey:
            return set()

    def _save_to_s3(self, dates: Set[str]):
        body = json.dumps({d: {} for d in sorted(dates)}).encode()
        self.s3.put_object(Bucket=self.bucket, Key=self.key, Body=body,
                           ContentType="application/json")
