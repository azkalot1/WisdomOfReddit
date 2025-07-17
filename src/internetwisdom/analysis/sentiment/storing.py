from typing import Protocol, Optional, List, Dict, Any, Set
from .state import DailyUpdate, DailySentimentBatch, SentimentResults, IncrementalDailyUpdate, ProcessingCheckpoint, ProcessingResult
from pathlib import Path
import json
import pickle
from datetime import datetime
from dataclasses import asdict
try:
    import fcntl  # For file locking on Unix systems
except ImportError:
    fcntl = None
try:
    import msvcrt  # For file locking on Windows
except ImportError:
    msvcrt = None

import logging
import platform
logger = logging.getLogger(__name__)

class SentimentStorage(Protocol):
    """Interface for storing processed sentiments"""
    
    def save_daily_update(self, update: DailyUpdate) -> None:
        """Save or update daily sentiment data"""
        ...
    
    def load_daily_batch(self, date: str) -> Optional[DailySentimentBatch]:
        """Load complete daily batch"""
        ...
    
    def get_submission_sentiments(self, submission_id: str) -> Optional[List[SentimentResults]]:
        """Get sentiments for a specific submission"""
        ...
    
    def is_submission_processed(self, submission_id: str) -> bool:
        """Check if submission has been processed"""
        ...
    
    def get_processed_dates(self) -> List[str]:
        """Get list of dates that have been processed"""
        ...



class LocalFileSentimentStorage:
    """Local file-based sentiment storage"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.daily_batches_path = self.base_path / "daily_batches"
        self.daily_batches_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_path / "submission_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load submission -> date mapping index"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.submission_index = json.load(f)
        else:
            self.submission_index = {}
    
    def _save_index(self):
        """Save submission index"""
        with open(self.index_path, 'w') as f:
            json.dump(self.submission_index, f, indent=2)
    
    def save_daily_update(self, update: DailyUpdate) -> None:
        """Save or update daily sentiment data"""
        # Load existing batch if it exists
        existing_batch = self.load_daily_batch(update.date)
        
        if existing_batch:
            # Update existing batch
            for submission_id, result in update.submission_results.items():
                existing_batch.submission_sentiments[submission_id] = result.sentiments
                self.submission_index[submission_id] = update.date
            
            existing_batch.last_updated = update.update_timestamp
            existing_batch.total_submissions = len(existing_batch.submission_sentiments)
            existing_batch.total_extractions = sum(
                len(sentiments) for sentiments in existing_batch.submission_sentiments.values()
            )
            
            # Update metadata
            if update.metadata:
                existing_batch.processing_metadata.update(update.metadata)
            
            batch_to_save = existing_batch
        else:
            # Create new batch
            submission_sentiments = {
                submission_id: result.sentiments 
                for submission_id, result in update.submission_results.items()
            }
            
            batch_to_save = DailySentimentBatch(
                date=update.date,
                submission_sentiments=submission_sentiments,
                processing_metadata=update.metadata or {},
                last_updated=update.update_timestamp,
                total_submissions=len(submission_sentiments),
                total_extractions=sum(len(sentiments) for sentiments in submission_sentiments.values())
            )
            
            # Update index
            for submission_id in submission_sentiments.keys():
                self.submission_index[submission_id] = update.date
        
        # Save batch file
        batch_file = self.daily_batches_path / f"{update.date}_sentiments.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_to_save, f)
        
        # Save index
        self._save_index()
        
        logger.info(f"✅ Updated daily batch for {update.date}: "
                   f"{len(update.submission_results)} submissions")
    
    def load_daily_batch(self, date: str) -> Optional[DailySentimentBatch]:
        """Load complete daily batch"""
        batch_file = self.daily_batches_path / f"{date}_sentiments.pkl"
        if not batch_file.exists():
            return None
        
        try:
            with open(batch_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load daily batch for {date}: {e}")
            return None
    
    def get_submission_sentiments(self, submission_id: str) -> Optional[List[SentimentResults]]:
        """Get sentiments for a specific submission"""
        if submission_id not in self.submission_index:
            return None
        
        date = self.submission_index[submission_id]
        batch = self.load_daily_batch(date)
        
        if batch and submission_id in batch.submission_sentiments:
            return batch.submission_sentiments[submission_id]
        
        return None
    
    def is_submission_processed(self, submission_id: str) -> bool:
        """Check if submission has been processed"""
        return submission_id in self.submission_index
    
    def get_processed_dates(self) -> List[str]:
        """Get list of dates that have been processed"""
        dates = []
        for file_path in self.base_path.glob("*_sentiments.pkl"):
            date = file_path.stem.replace("_sentiments", "")
            dates.append(date)
        return sorted(dates)
    
    def get_unprocessed_submissions(self, date: str, all_submission_ids: List[str]) -> List[str]:
        """Get submissions for a date that haven't been processed yet"""
        batch = self.load_daily_batch(date)
        if not batch:
            return all_submission_ids
        
        processed_ids = set(batch.submission_sentiments.keys())
        return [sub_id for sub_id in all_submission_ids if sub_id not in processed_ids]


class IncrementalSentimentStorage:
    """
    Storage that saves submissions incrementally as they're processed.
    
    Structure:
    base_path/
    ├── checkpoints/
    │   ├── 20200301_checkpoint.json     # Processing progress tracking
    │   └── 20200302_checkpoint.json
    ├── incremental/
    │   ├── 20200301/
    │   │   ├── abc123.pkl              # Individual processed submissions
    │   │   ├── def456.pkl
    │   │   └── ghi789.pkl
    │   └── 20200302/
    ├── daily_batches/
    │   ├── 20200301_sentiments.pkl     # Final consolidated daily files
    │   └── 20200302_sentiments.pkl
    └── submission_index.json           # Fast lookup index
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoints_path = self.base_path / "checkpoints"
        self.incremental_path = self.base_path / "incremental"
        self.daily_batches_path = self.base_path / "daily_batches"
        
        for path in [self.checkpoints_path, self.incremental_path, self.daily_batches_path]:
            path.mkdir(exist_ok=True)
        
        self.index_path = self.base_path / "submission_index.json"
        self._load_index()
        
        logger.info(f"Initialized IncrementalSentimentStorage at {self.base_path}")
        logger.info(f"  Checkpoints: {self.checkpoints_path}")
        logger.info(f"  Incremental: {self.incremental_path}")
        logger.info(f"  Daily batches: {self.daily_batches_path}")
    
    def _load_index(self):
        """Load submission -> date mapping index"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    self.submission_index = json.load(f)
                logger.debug(f"Loaded index with {len(self.submission_index)} entries")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self.submission_index = {}
        else:
            self.submission_index = {}
    
    def _save_index(self):
        """Save submission index with file locking"""
        try:
            self._atomic_write_json(self.index_path, self.submission_index)
            logger.debug(f"Saved index with {len(self.submission_index)} entries")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _atomic_write_json(self, file_path: Path, data: Dict):
        """Atomically write JSON with file locking"""
        temp_path = file_path.with_suffix('.tmp')
        
        try:
            with open(temp_path, 'w') as f:
                # Lock file during write
                if platform.system() == 'Windows':
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    except ImportError:
                        logger.warning("File locking not available")
                else:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    except ImportError:
                        logger.warning("File locking not available")
                
                json.dump(data, f, indent=2, default=str)
            
            # Atomic move
            temp_path.replace(file_path)
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def load_checkpoint(self, date: str) -> Optional[ProcessingCheckpoint]:
        """Load processing checkpoint for a date"""
        checkpoint_file = self.checkpoints_path / f"{date}_checkpoint.json"
        
        if not checkpoint_file.exists():
            logger.debug(f"No checkpoint found for {date}")
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Convert sets back from lists
            data['processed_submissions'] = set(data['processed_submissions'])
            data['failed_submissions'] = set(data['failed_submissions'])
            data['last_checkpoint_time'] = datetime.fromisoformat(data['last_checkpoint_time'])
            
            logger.debug(f"Loaded checkpoint for {date}: {len(data['processed_submissions'])} processed")
            return ProcessingCheckpoint(**data)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {date}: {e}")
            return None
    
    def save_checkpoint(self, checkpoint: ProcessingCheckpoint):
        """Save processing checkpoint"""
        checkpoint_file = self.checkpoints_path / f"{checkpoint.date}_checkpoint.json"
        
        # Convert to serializable format
        data = asdict(checkpoint)
        data['processed_submissions'] = list(checkpoint.processed_submissions)
        data['failed_submissions'] = list(checkpoint.failed_submissions)
        data['last_checkpoint_time'] = checkpoint.last_checkpoint_time.isoformat()
        
        try:
            self._atomic_write_json(checkpoint_file, data)
            logger.debug(f"Saved checkpoint for {checkpoint.date}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {checkpoint.date}: {e}")
            raise
    
    def save_submission_result(self, date: str, submission_id: str, 
                             result: ProcessingResult) -> None:
        """Save individual submission result immediately"""
        # Create date directory
        date_dir = self.incremental_path / date
        date_dir.mkdir(exist_ok=True)
        
        # Save individual result
        result_file = date_dir / f"{submission_id}.pkl"
        try:
            with open(result_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Update index
            self.submission_index[submission_id] = date
            self._save_index()
            
            logger.debug(f"Saved result for {submission_id} in {date}")
            
        except Exception as e:
            logger.error(f"Failed to save result for {submission_id}: {e}")
            raise
    
    def load_submission_result(self, date: str, submission_id: str) -> Optional[ProcessingResult]:
        """Load individual submission result"""
        result_file = self.incremental_path / date / f"{submission_id}.pkl"
        
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load result for {submission_id}: {e}")
            return None
    
    def get_processed_submissions_for_date(self, date: str) -> Set[str]:
        """Get set of submission IDs that have been processed for a date"""
        # Check checkpoint first
        checkpoint = self.load_checkpoint(date)
        if checkpoint:
            return checkpoint.processed_submissions.copy()
        
        # Fall back to scanning incremental directory
        date_dir = self.incremental_path / date
        if not date_dir.exists():
            return set()
        
        processed = set()
        for result_file in date_dir.glob("*.pkl"):
            processed.add(result_file.stem)
        
        return processed
    
    def get_failed_submissions_for_date(self, date: str) -> Set[str]:
        """Get set of submission IDs that failed processing for a date"""
        checkpoint = self.load_checkpoint(date)
        if checkpoint:
            return checkpoint.failed_submissions.copy()
        return set()
    
    def consolidate_daily_results(self, date: str) -> Optional[DailySentimentBatch]:
        """Consolidate incremental results into final daily batch"""
        logger.info(f"Consolidating results for {date}")
        
        date_dir = self.incremental_path / date
        if not date_dir.exists():
            logger.warning(f"No incremental results found for {date}")
            return None
        
        # Load all individual results
        submission_sentiments = {}
        total_extractions = 0
        processing_metadata = {}
        successful_count = 0
        failed_count = 0
        
        # Get list of all result files
        result_files = list(date_dir.glob("*.pkl"))
        logger.info(f"Found {len(result_files)} result files for {date}")
        
        for result_file in result_files:
            submission_id = result_file.stem
            
            try:
                result = self.load_submission_result(date, submission_id)
                if result:
                    if not result.error:
                        submission_sentiments[submission_id] = result.sentiments
                        total_extractions += result.extraction_count
                        successful_count += 1
                    else:
                        failed_count += 1
                        logger.debug(f"Skipping failed result for {submission_id}: {result.error}")
                else:
                    logger.warning(f"Could not load result for {submission_id}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to load {result_file}: {e}")
                failed_count += 1
        
        if not submission_sentiments:
            logger.warning(f"No valid results found for {date}")
            return None
        
        # Get checkpoint info for metadata
        checkpoint = self.load_checkpoint(date)
        if checkpoint:
            processing_metadata.update({
                'processing_sessions': checkpoint.metadata or {},
                'total_submissions_found': checkpoint.total_submissions_found,
                'checkpoint_last_updated': checkpoint.last_checkpoint_time.isoformat()
            })
        
        processing_metadata.update({
            'consolidation_timestamp': datetime.now().isoformat(),
            'successful_results': successful_count,
            'failed_results': failed_count,
            'total_result_files': len(result_files)
        })
        
        # Create consolidated batch
        daily_batch = DailySentimentBatch(
            date=date,
            submission_sentiments=submission_sentiments,
            processing_metadata=processing_metadata,
            last_updated=datetime.now(),
            total_submissions=len(submission_sentiments),
            total_extractions=total_extractions
        )
        
        # Save consolidated batch to the correct directory
        batch_file = self.daily_batches_path / f"{date}_sentiments.pkl"  # FIX: Use daily_batches_path
        try:
            with open(batch_file, 'wb') as f:
                pickle.dump(daily_batch, f)
            
            logger.info(f"✅ Consolidated {len(submission_sentiments)} submissions for {date}")
            logger.info(f"   Saved to: {batch_file}")
            logger.info(f"   Total extractions: {total_extractions}")
            
        except Exception as e:
            logger.error(f"Failed to save consolidated batch for {date}: {e}")
            raise
        
        return daily_batch
    
    def cleanup_incremental_files(self, date: str):
        """Clean up incremental files after consolidation"""
        date_dir = self.incremental_path / date
        if not date_dir.exists():
            logger.info(f"No incremental directory to clean up for {date}")
            return
        
        try:
            # Count files before cleanup
            file_count = len(list(date_dir.glob("*.pkl")))
            
            # Remove all pickle files
            for file in date_dir.glob("*.pkl"):
                file.unlink()
            
            # Remove directory if empty
            try:
                date_dir.rmdir()
                logger.info(f"✅ Cleaned up {file_count} incremental files for {date}")
            except OSError:
                # Directory not empty (might have other files)
                logger.info(f"✅ Cleaned up {file_count} incremental files for {date} (directory kept)")
                
        except Exception as e:
            logger.error(f"Failed to cleanup incremental files for {date}: {e}")
            raise
    
    def get_unprocessed_submissions(self, date: str, all_submission_ids: List[str]) -> List[str]:
        """Get submissions that haven't been processed yet"""
        processed = self.get_processed_submissions_for_date(date)
        failed = self.get_failed_submissions_for_date(date)
        
        # Don't retry failed submissions unless explicitly requested
        already_handled = processed.union(failed)
        
        unprocessed = [sub_id for sub_id in all_submission_ids if sub_id not in already_handled]
        
        logger.debug(f"Unprocessed submissions for {date}: {len(unprocessed)}/{len(all_submission_ids)}")
        return unprocessed
    
    def get_processing_summary(self, date: str, all_submission_ids: List[str]) -> Dict[str, Any]:
        """Get detailed processing summary for a date"""
        processed = self.get_processed_submissions_for_date(date)
        failed = self.get_failed_submissions_for_date(date)
        unprocessed = set(all_submission_ids) - processed - failed
        
        checkpoint = self.load_checkpoint(date)
        
        # Check for consolidated batch
        consolidated_batch_exists = (self.daily_batches_path / f"{date}_sentiments.pkl").exists()
        incremental_files_exist = (self.incremental_path / date).exists()
        
        summary = {
            'date': date,
            'total_available': len(all_submission_ids),
            'processed': len(processed),
            'failed': len(failed),
            'unprocessed': len(unprocessed),
            'processing_complete': len(unprocessed) == 0,
            'success_rate': len(processed) / (len(processed) + len(failed)) if (processed or failed) else 0,
            'last_checkpoint': checkpoint.last_checkpoint_time.isoformat() if checkpoint else None,
            'has_incremental_files': incremental_files_exist,
            'has_consolidated_batch': consolidated_batch_exists,
            'storage_paths': {
                'incremental_dir': str(self.incremental_path / date),
                'consolidated_file': str(self.daily_batches_path / f"{date}_sentiments.pkl"),
                'checkpoint_file': str(self.checkpoints_path / f"{date}_checkpoint.json")
            }
        }
        
        # Add file counts if directories exist
        if incremental_files_exist:
            incremental_file_count = len(list((self.incremental_path / date).glob("*.pkl")))
            summary['incremental_file_count'] = incremental_file_count
        
        return summary
    
    # Implement the original interface methods for compatibility
    def save_daily_update(self, update: DailyUpdate) -> None:
        """Save daily update (compatibility method)"""
        # This is now handled incrementally, but we can still support batch updates
        for submission_id, result in update.submission_results.items():
            self.save_submission_result(update.date, submission_id, result)
    
    def load_daily_batch(self, date: str) -> Optional[DailySentimentBatch]:
        """Load daily batch (try consolidated first, then build from incremental)"""
        # Try consolidated batch first
        batch_file = self.daily_batches_path / f"{date}_sentiments.pkl"  # FIX: Use daily_batches_path
        if batch_file.exists():
            try:
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f)
                logger.debug(f"Loaded consolidated batch for {date}: {batch.total_submissions} submissions")
                return batch
            except Exception as e:
                logger.error(f"Failed to load consolidated batch for {date}: {e}")
        
        # Build from incremental files if no consolidated batch
        logger.debug(f"No consolidated batch found for {date}, building from incremental files")
        return self.consolidate_daily_results(date)
    
    def get_submission_sentiments(self, submission_id: str) -> Optional[List[SentimentResults]]:
        """Get sentiments for a specific submission"""
        if submission_id not in self.submission_index:
            return None
        
        date = self.submission_index[submission_id]
        
        # Try incremental file first
        result = self.load_submission_result(date, submission_id)
        if result and not result.error:
            return result.sentiments
        
        # Fall back to consolidated batch
        batch = self.load_daily_batch(date)
        if batch and submission_id in batch.submission_sentiments:
            return batch.submission_sentiments[submission_id]
        
        return None
    
    def is_submission_processed(self, submission_id: str) -> bool:
        """Check if submission has been processed"""
        return submission_id in self.submission_index
    
    def list_dates_with_data(self) -> Dict[str, Dict[str, Any]]:
        """List all dates that have any kind of processing data"""
        dates_info = {}
        
        # Check incremental directories
        if self.incremental_path.exists():
            for date_dir in self.incremental_path.iterdir():
                if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
                    date = date_dir.name
                    file_count = len(list(date_dir.glob("*.pkl")))
                    if file_count > 0:
                        dates_info[date] = {
                            'has_incremental': True,
                            'incremental_file_count': file_count,
                            'has_consolidated': False,
                            'has_checkpoint': False
                        }
        
        # Check consolidated batches
        if self.daily_batches_path.exists():
            for batch_file in self.daily_batches_path.glob("*_sentiments.pkl"):
                date = batch_file.stem.replace("_sentiments", "")
                if date not in dates_info:
                    dates_info[date] = {
                        'has_incremental': False,
                        'incremental_file_count': 0,
                        'has_consolidated': True,
                        'has_checkpoint': False
                    }
                else:
                    dates_info[date]['has_consolidated'] = True
        
        # Check checkpoints
        if self.checkpoints_path.exists():
            for checkpoint_file in self.checkpoints_path.glob("*_checkpoint.json"):
                date = checkpoint_file.stem.replace("_checkpoint", "")
                if date not in dates_info:
                    dates_info[date] = {
                        'has_incremental': False,
                        'incremental_file_count': 0,
                        'has_consolidated': False,
                        'has_checkpoint': True
                    }
                else:
                    dates_info[date]['has_checkpoint'] = True
        
        return dates_info
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get overall storage statistics"""
        dates_info = self.list_dates_with_data()
        
        total_incremental_files = sum(
            info['incremental_file_count'] for info in dates_info.values()
        )
        
        consolidated_count = sum(
            1 for info in dates_info.values() if info['has_consolidated']
        )
        
        checkpoint_count = sum(
            1 for info in dates_info.values() if info['has_checkpoint']
        )
        
        return {
            'total_dates_with_data': len(dates_info),
            'total_incremental_files': total_incremental_files,
            'consolidated_batches': consolidated_count,
            'active_checkpoints': checkpoint_count,
            'total_submissions_indexed': len(self.submission_index),
            'storage_paths': {
                'base_path': str(self.base_path),
                'checkpoints': str(self.checkpoints_path),
                'incremental': str(self.incremental_path),
                'daily_batches': str(self.daily_batches_path)
            }
        }