from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
from .reddit import RedditCommentFormatter, PrawRedditClient
from .data import JsonFileStorage, FilePathManager

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of processing a single submission"""
    submission_id: str
    success: bool
    file_path: Optional[Path] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    error_message: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None
    processing_time_seconds: Optional[float] = None

class SubmissionProcessor:
    """
    Processes individual Reddit submissions through the complete pipeline
    
    Single Responsibility: Complete processing pipeline for ONE submission
    Coordinates: fetching → formatting → path generation → saving
    """
    
    def __init__(self, 
                 reddit_client: PrawRedditClient,
                 formatter: RedditCommentFormatter,
                 storage: JsonFileStorage,
                 path_manager: FilePathManager,
                 comment_limit: int = 1000):
        """
        Initialize submission processor with all required dependencies
        
        Args:
            reddit_client: PrawRedditClient instance
            formatter: RedditCommentFormatter instance
            storage: JsonFileStorage instance
            path_manager: FilePathManager instance
            comment_limit: Maximum comments to fetch per submission
        """
        self.reddit_client = reddit_client
        self.formatter = formatter
        self.storage = storage
        self.path_manager = path_manager
        self.comment_limit = comment_limit
    
    def process_submission(
            self, 
            submission_id: str, 
            force_update: bool = False,
            include_metadata: bool = True) -> ProcessingResult:
        """
        Process a single Reddit submission through the complete pipeline
        
        Args:
            submission_id: Reddit submission ID to process
            force_update: If True, overwrite existing files
            include_metadata: Whether to include metadata in JSON output
            
        Returns:
            ProcessingResult with outcome and statistics
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting processing for submission {submission_id}")
            
            # Step 1: Fetch submission data
            logger.debug(f"Fetching submission data for {submission_id}")
            submission_data = self.reddit_client.get_submission(submission_id)
            
            if not submission_data:
                return ProcessingResult(
                    submission_id=submission_id,
                    success=False,
                    error_message="Failed to fetch submission data"
                )
            
            # Step 2: Generate file path early to check if file exists
            formatted_temp = {
                'submission_id': submission_data.get('id', submission_id),
                'date': self._format_date(submission_data.get('created_utc', 0)),
                'subreddit': submission_data.get('subreddit', '')
            }
            
            file_path = self.path_manager.get_submission_file_path(
                formatted_temp['submission_id'],
                formatted_temp['date'], 
                formatted_temp['subreddit']
            )
            
            # Step 3: Check if file exists and handle accordingly
            if not force_update and self.storage.exists(file_path):
                logger.info(f"File already exists and force_update=False: {file_path}")
                return ProcessingResult(
                    submission_id=submission_id,
                    success=True,
                    file_path=file_path,
                    skipped=True,
                    skip_reason="File already exists",
                    processing_time_seconds=(datetime.utcnow() - start_time).total_seconds()
                )
            
            # Step 4: Fetch comments
            logger.debug(f"Fetching comments for {submission_id} (limit: {self.comment_limit})")
            raw_comments = self.reddit_client.get_comments(submission_id, limit=self.comment_limit)
            
            # Step 5: Format data
            logger.debug(f"Formatting data for {submission_id}")
            formatted_data = self.formatter.format_submission_data(
                submission_data, 
                raw_comments,
                include_metadata=include_metadata
            )
            
            # Step 6: Ensure directory exists
            self.path_manager.ensure_directory_exists(file_path)
            
            # Step 7: Save to file
            logger.debug(f"Saving data to {file_path}")
            save_success = self.storage.save(formatted_data, file_path)
            
            if not save_success:
                return ProcessingResult(
                    submission_id=submission_id,
                    success=False,
                    file_path=file_path,
                    error_message="Failed to save JSON file"
                )
            
            # Step 8: Extract statistics
            statistics = self._extract_processing_statistics(formatted_data, raw_comments)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Successfully processed {submission_id} in {processing_time:.2f}s → {file_path}")
            
            return ProcessingResult(
                submission_id=submission_id,
                success=True,
                file_path=file_path,
                statistics=statistics,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Failed to process submission {submission_id}: {e}")
            
            return ProcessingResult(
                submission_id=submission_id,
                success=False,
                error_message=str(e),
                processing_time_seconds=processing_time
            )
    
    def _format_date(self, created_utc: float) -> str:
        """Format Unix timestamp to ISO format"""
        if not created_utc:
            return datetime.utcnow().isoformat()[:19]
        
        dt = datetime.utcfromtimestamp(created_utc)
        return dt.isoformat()[:19]
    
    def _extract_processing_statistics(
            self,
            formatted_data: Dict[str, Any],
            raw_comments: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
        """Extract key statistics from processing"""
        stats = {
            'submission_id': formatted_data.get('submission_id'),
            'subreddit': formatted_data.get('subreddit'),
            'title_length': len(formatted_data.get('title', '')),
            'raw_comment_count': len(raw_comments),
            'formatted_comment_count': len(formatted_data.get('comments', [])),
            'processing_timestamp': datetime.utcnow().isoformat()
        }
        
        # Add metadata statistics if available
        if 'metadata' in formatted_data:
            metadata = formatted_data['metadata']
            if 'statistics' in metadata:
                stats.update({
                    'total_score': metadata['statistics'].get('score_statistics', {}).get('total_score', 0),
                    'avg_score': metadata['statistics'].get('score_statistics', {}).get('average_score', 0),
                    'top_level_comments': metadata['statistics'].get('comment_breakdown', {}).get('top_level_comments', 0),
                    'total_replies': metadata['statistics'].get('comment_breakdown', {}).get('total_replies', 0)
                })
        
        return stats
    
    def process_multiple_submissions(
            self, 
            submission_ids: List[str],
            force_update: bool = False,
            include_metadata: bool = True
        ) -> List[ProcessingResult]:
        """
        Process multiple submissions
        
        Args:
            submission_ids: List of Reddit submission IDs
            force_update: If True, overwrite existing files
            include_metadata: Whether to include metadata in JSON output
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        logger.info(f"Processing {len(submission_ids)} submissions")
        
        for i, submission_id in enumerate(submission_ids, 1):
            logger.info(f"Processing {i}/{len(submission_ids)}: {submission_id}")
            
            result = self.process_submission(
                submission_id, 
                force_update=force_update,
                include_metadata=include_metadata
            )
            
            results.append(result)
            
            # Log progress
            if result.success:
                if result.skipped:
                    logger.info(f"  → Skipped ({result.skip_reason})")
                else:
                    logger.info(f"  → Success ({result.processing_time_seconds:.2f}s)")
            else:
                logger.error(f"  → Failed: {result.error_message}")
        
        # Summary statistics
        successful = len([r for r in results if r.success and not r.skipped])
        skipped = len([r for r in results if r.skipped])
        failed = len([r for r in results if not r.success])
        
        logger.info(f"Batch processing complete: {successful} processed, {skipped} skipped, {failed} failed")
        
        return results