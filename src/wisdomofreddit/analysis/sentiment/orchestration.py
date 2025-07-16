from typing import Dict, List, Any, Optional
from .loading import ContentLoader
from .processing import SentimentProcessor
from .storing import SentimentStorage, IncrementalSentimentStorage
from .state import DailyUpdate, DailySentimentBatch, ProcessingCheckpoint
from datetime import datetime
from abc import ABC, abstractmethod
from .prescoring import SentimentPrescorer
import uuid
import asyncio
import logging
logger = logging.getLogger(__name__)


class BaseSentimentOrchestrator(ABC):
    """
    Base class for sentiment processing orchestrators with cost estimation.
    
    Provides common functionality for cost estimation and basic orchestration.
    """
    
    def __init__(self, 
                 content_loader: ContentLoader,
                 sentiment_processor: SentimentProcessor,
                 sentiment_storage: SentimentStorage,
                 sentiment_prescorer: SentimentPrescorer
                 ):
        self.content_loader = content_loader
        self.sentiment_processor = sentiment_processor
        self.sentiment_storage = sentiment_storage
        self.sentiment_prescorer = sentiment_prescorer

    
    def estimate_cost_for_day(self, date: str, only_unprocessed: bool = True) -> Dict[str, Any]:
        """
        Estimate processing cost for a specific day.
        
        Args:
            date: Date to estimate (format: YYYYMMDD)
            only_unprocessed: If True, only estimate cost for unprocessed submissions
            
        Returns:
            Dictionary with cost estimates and token information
        """
        logger.info(f"Estimating cost for {date} (only_unprocessed={only_unprocessed})")
        
        # Get submissions for the date
        all_submission_ids = self.content_loader.get_submissions_for_date(date)
        
        if not all_submission_ids:
            return {
                'date': date,
                'total_submissions': 0,
                'submissions_to_estimate': 0,
                'error': 'No submissions found for date'
            }
        
        # Determine which submissions to estimate
        if only_unprocessed:
            submission_ids_to_estimate = self._get_unprocessed_submissions_for_date(date, all_submission_ids)
            estimation_type = 'unprocessed_only'
        else:
            submission_ids_to_estimate = all_submission_ids
            estimation_type = 'all_submissions'
        
        if not submission_ids_to_estimate:
            return {
                'date': date,
                'total_submissions': len(all_submission_ids),
                'submissions_to_estimate': 0,
                'estimation_type': estimation_type,
                'message': 'No submissions to estimate (all already processed)' if only_unprocessed else 'No submissions found',
                'cost_estimate': {
                    'total_cost': '$0.00',
                    'input_cost': '$0.00',
                    'output_cost': '$0.00'
                }
            }
        
        logger.info(f"Loading {len(submission_ids_to_estimate)} submissions for cost estimation")
        
        # Load content for submissions to estimate
        contents = []
        failed_to_load = []
        
        for submission_id in submission_ids_to_estimate:
            content = self.content_loader.load_submission_content(date, submission_id)
            if content:
                contents.append(content)
            else:
                failed_to_load.append(submission_id)
        
        if failed_to_load:
            logger.warning(f"Failed to load {len(failed_to_load)} submissions for estimation")
        
        if not contents:
            return {
                'date': date,
                'total_submissions': len(all_submission_ids),
                'submissions_to_estimate': len(submission_ids_to_estimate),
                'estimation_type': estimation_type,
                'error': 'Could not load any content for estimation',
                'failed_to_load': len(failed_to_load)
            }
        
        # Get the extractor from the processor to use its estimation methods
        extractor = self.sentiment_processor.extractor
        
        # Estimate cost using the extractor's batch estimation
        cost_estimate = extractor.estimate_batch_tokens(contents)
        
        # Add our additional metadata
        result = {
            'date': date,
            'estimation_type': estimation_type,
            'total_submissions_available': len(all_submission_ids),
            'submissions_to_estimate': len(submission_ids_to_estimate),
            'submissions_loaded_for_estimation': len(contents),
            'failed_to_load': len(failed_to_load),
            'estimation_timestamp': datetime.now().isoformat(),
            
            # Cost and token estimates from extractor
            **cost_estimate,
            
            # Additional calculated fields
            'avg_cost_per_submission': self._parse_cost(cost_estimate['cost_estimate']['total_cost']) / len(contents) if contents else 0,
            'processing_estimates': {
                'estimated_processing_time_minutes': len(contents) * cost_estimate['token_summary']['avg_extractions_per_submission'] / 8,  # Rough estimate: 8 extractions per minute
                'estimated_api_calls': cost_estimate['total_extractions']
            }
        }
        
        logger.info(f"Cost estimate for {date}: {cost_estimate['cost_estimate']['total_cost']} "
                   f"for {len(contents)} submissions ({cost_estimate['total_extractions']} extractions)")
        
        return result
    
    def estimate_cost_for_date_range(self, start_date: str, end_date: str, 
                                   only_unprocessed: bool = True,
                                   sample_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Estimate processing cost for a date range.
        
        Args:
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            only_unprocessed: If True, only estimate cost for unprocessed submissions
            sample_days: If provided, estimate based on sample of days and extrapolate
            
        Returns:
            Dictionary with cost estimates for the date range
        """
        logger.info(f"Estimating cost for date range {start_date} to {end_date} "
                   f"(only_unprocessed={only_unprocessed}, sample_days={sample_days})")
        
        # Get available dates in range
        available_dates = self.content_loader.get_available_dates()
        dates_in_range = [
            date for date in available_dates 
            if start_date <= date <= end_date
        ]
        
        if not dates_in_range:
            return {
                'start_date': start_date,
                'end_date': end_date,
                'total_dates': 0,
                'error': 'No dates found in range'
            }
        
        # Determine which dates to sample for estimation
        if sample_days and sample_days < len(dates_in_range):
            # Sample evenly across the range
            step = len(dates_in_range) // sample_days
            sampled_dates = dates_in_range[::step][:sample_days]
            is_sampled = True
        else:
            sampled_dates = dates_in_range
            is_sampled = False
        
        logger.info(f"Estimating based on {len(sampled_dates)} dates "
                   f"{'(sampled)' if is_sampled else '(all dates)'}")
        
        # Estimate cost for each sampled date
        daily_estimates = {}
        total_submissions = 0
        total_submissions_to_estimate = 0
        total_submissions_loaded = 0
        total_failed_to_load = 0
        total_extractions = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        
        for date in sampled_dates:
            try:
                daily_estimate = self.estimate_cost_for_day(date, only_unprocessed)
                daily_estimates[date] = daily_estimate
                
                if 'error' not in daily_estimate:
                    total_submissions += daily_estimate.get('total_submissions_available', 0)
                    total_submissions_to_estimate += daily_estimate.get('submissions_to_estimate', 0)
                    total_submissions_loaded += daily_estimate.get('submissions_loaded_for_estimation', 0)
                    total_failed_to_load += daily_estimate.get('failed_to_load', 0)
                    total_extractions += daily_estimate.get('total_extractions', 0)
                    total_input_tokens += daily_estimate.get('total_input_tokens', 0)
                    total_output_tokens += daily_estimate.get('total_output_tokens', 0)
                    total_cost += self._parse_cost(daily_estimate.get('cost_estimate', {}).get('total_cost', '$0.00'))
                
            except Exception as e:
                logger.error(f"Failed to estimate cost for {date}: {e}")
                daily_estimates[date] = {'error': str(e)}
        
        # Calculate averages and extrapolate if sampling
        if total_submissions_loaded > 0:
            avg_submissions_per_day = total_submissions / len(sampled_dates)
            avg_submissions_to_estimate_per_day = total_submissions_to_estimate / len(sampled_dates)
            avg_cost_per_day = total_cost / len(sampled_dates)
            avg_extractions_per_day = total_extractions / len(sampled_dates)
            
            if is_sampled:
                # Extrapolate to full range
                extrapolated_total_cost = avg_cost_per_day * len(dates_in_range)
                extrapolated_total_submissions = avg_submissions_per_day * len(dates_in_range)
                extrapolated_total_submissions_to_estimate = avg_submissions_to_estimate_per_day * len(dates_in_range)
                extrapolated_total_extractions = avg_extractions_per_day * len(dates_in_range)
            else:
                extrapolated_total_cost = total_cost
                extrapolated_total_submissions = total_submissions
                extrapolated_total_submissions_to_estimate = total_submissions_to_estimate
                extrapolated_total_extractions = total_extractions
        else:
            avg_submissions_per_day = 0
            avg_submissions_to_estimate_per_day = 0
            avg_cost_per_day = 0
            avg_extractions_per_day = 0
            extrapolated_total_cost = 0
            extrapolated_total_submissions = 0
            extrapolated_total_submissions_to_estimate = 0
            extrapolated_total_extractions = 0
        
        result = {
            'start_date': start_date,
            'end_date': end_date,
            'estimation_type': 'unprocessed_only' if only_unprocessed else 'all_submissions',
            'total_dates_in_range': len(dates_in_range),
            'dates_sampled_for_estimation': len(sampled_dates),
            'is_sampled_estimate': is_sampled,
            'estimation_timestamp': datetime.now().isoformat(),
            
            # Sampled data summary
            'sampled_data': {
                'total_submissions': total_submissions,
                'total_submissions_to_estimate': total_submissions_to_estimate,
                'total_submissions_loaded': total_submissions_loaded,
                'total_failed_to_load': total_failed_to_load,
                'total_extractions': total_extractions,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'total_cost': f"${total_cost:.2f}"
            },
            
            # Averages
            'daily_averages': {
                'avg_submissions_per_day': round(avg_submissions_per_day, 1),
                'avg_submissions_to_estimate_per_day': round(avg_submissions_to_estimate_per_day, 1),
                'avg_extractions_per_day': round(avg_extractions_per_day, 1),
                'avg_cost_per_day': f"${avg_cost_per_day:.2f}"
            },
            
            # Extrapolated totals (for full date range)
            'range_estimates': {
                'estimated_total_submissions': round(extrapolated_total_submissions),
                'estimated_total_submissions_to_process': round(extrapolated_total_submissions_to_estimate),
                'estimated_total_extractions': round(extrapolated_total_extractions),
                'estimated_total_cost': f"${extrapolated_total_cost:.2f}",
                'estimated_processing_time_hours': round(extrapolated_total_extractions / 480, 1)  # Assuming 8 extractions/minute
            },
            
            # Daily breakdown
            'daily_estimates': daily_estimates
        }
        
        logger.info(f"Date range cost estimate: {result['range_estimates']['estimated_total_cost']} "
                   f"for {result['range_estimates']['estimated_total_submissions_to_process']} submissions "
                   f"across {len(dates_in_range)} dates")
        
        return result
    
    def _parse_cost(self, cost_string: str) -> float:
        """Parse cost string like '$1.23' to float 1.23"""
        if isinstance(cost_string, (int, float)):
            return float(cost_string)
        
        try:
            return float(cost_string.replace('$', '').replace(',', ''))
        except (ValueError, AttributeError):
            return 0.0
    
    def get_cost_summary_for_processed_data(self, date: str) -> Dict[str, Any]:
        """
        Get cost summary for already processed data (retrospective analysis).
        
        Args:
            date: Date to analyze
            
        Returns:
            Dictionary with cost analysis of processed data
        """
        daily_batch = self.sentiment_storage.load_daily_batch(date)
        
        if not daily_batch:
            return {
                'date': date,
                'error': 'No processed data found for date'
            }
        
        # Get the extractor for cost calculation
        extractor = self.sentiment_processor.extractor
        
        # Calculate retrospective costs based on processed data
        total_extractions = daily_batch.total_extractions
        
        # Estimate tokens based on average (since we don't store token counts)
        # This is an approximation - for exact costs, you'd need to store token counts during processing
        estimated_avg_tokens_per_extraction = 1000  # Adjust based on your typical extraction size
        estimated_total_input_tokens = total_extractions * estimated_avg_tokens_per_extraction
        estimated_total_output_tokens = total_extractions * 300  # Typical output size
        
        # Calculate costs
        input_cost = (estimated_total_input_tokens / 1_000_000) * extractor._get_input_token_cost()
        output_cost = (estimated_total_output_tokens / 1_000_000) * extractor._get_output_token_cost()
        total_cost = input_cost + output_cost
        
        return {
            'date': date,
            'processed_submissions': daily_batch.total_submissions,
            'total_extractions': total_extractions,
            'last_updated': daily_batch.last_updated.isoformat(),
            'estimated_costs': {
                'estimated_input_tokens': estimated_total_input_tokens,
                'estimated_output_tokens': estimated_total_output_tokens,
                'estimated_input_cost': f"${input_cost:.2f}",
                'estimated_output_cost': f"${output_cost:.2f}",
                'estimated_total_cost': f"${total_cost:.2f}",
                'cost_per_submission': f"${total_cost / daily_batch.total_submissions:.4f}",
                'cost_per_extraction': f"${total_cost / total_extractions:.4f}"
            },
            'note': 'Costs are estimated based on average token usage. For exact costs, token usage should be tracked during processing.'
        }

    
    @abstractmethod
    def _get_unprocessed_submissions_for_date(self, date: str, all_submission_ids: List[str]) -> List[str]:
        """Get unprocessed submissions for a date (implementation varies by storage type)"""
        pass
    
    @abstractmethod
    async def process_day(self, date: str, **kwargs) -> DailyUpdate:
        """Process a day (implementation varies by orchestrator type)"""
        pass
    
    @abstractmethod
    async def update_day(self, date: str, **kwargs) -> DailyUpdate:
        """Update a day (implementation varies by orchestrator type)"""
        pass
    
    
    def get_processing_summary(self, date: str) -> Dict[str, Any]:
        """
        Get a summary of processing status for a date.
        
        Args:
            date: Date to check
            
        Returns:
            Dictionary with processing summary
        """
        # Get available submissions
        available_submissions = self.content_loader.get_submissions_for_date(date)
        
        # Get processed submissions (implementation varies by storage type)
        unprocessed_submissions = self._get_unprocessed_submissions_for_date(date, available_submissions)
        processed_count = len(available_submissions) - len(unprocessed_submissions)
        
        # Try to get more detailed info from storage
        daily_batch = self.sentiment_storage.load_daily_batch(date)
        
        return {
            'date': date,
            'total_available': len(available_submissions),
            'total_processed': processed_count,
            'total_unprocessed': len(unprocessed_submissions),
            'processing_complete': len(unprocessed_submissions) == 0,
            'last_updated': daily_batch.last_updated.isoformat() if daily_batch else None,
            'total_extractions': daily_batch.total_extractions if daily_batch else 0,
            'unprocessed_sample': unprocessed_submissions[:10] if unprocessed_submissions else []
        }


class SentimentProcessingOrchestrator(BaseSentimentOrchestrator):
    """Standard sentiment processing orchestrator"""
    
    def _get_unprocessed_submissions_for_date(self, date: str, all_submission_ids: List[str]) -> List[str]:
        """Get unprocessed submissions using standard storage interface"""
        if hasattr(self.sentiment_storage, 'get_unprocessed_submissions'):
            return self.sentiment_storage.get_unprocessed_submissions(date, all_submission_ids)
        else:
            # Fallback: check each submission individually
            unprocessed = []
            for sub_id in all_submission_ids:
                if not self.sentiment_storage.is_submission_processed(sub_id):
                    unprocessed.append(sub_id)
            return unprocessed
    
    async def process_day(self, date: str, force_reprocess: bool = False) -> DailyUpdate:
        """Process all submissions for a specific day"""
        logger.info(f"Processing day: {date}")
        
        # Get all submissions for the date
        all_submission_ids = self.content_loader.get_submissions_for_date(date)
        
        if not force_reprocess:
            # Filter out already processed submissions
            unprocessed_ids = self._get_unprocessed_submissions_for_date(date, all_submission_ids)
            
            if not unprocessed_ids:
                logger.info(f"All submissions for {date} already processed")
                return DailyUpdate(
                    date=date,
                    submission_results={},
                    is_complete_day=True,
                    update_timestamp=datetime.now(),
                    metadata={'skipped_reason': 'already_processed'}
                )
            logger.info(f"Processing {len(unprocessed_ids)} submissions for {date}")
            submission_ids_to_process = unprocessed_ids
        else:
            submission_ids_to_process = all_submission_ids
        
        logger.info(f"Processing {len(submission_ids_to_process)} submissions for {date}")
        
        # Load content for unprocessed submissions
        contents = []
        skipped_count = 0
        for submission_id in submission_ids_to_process:
            content = self.content_loader.load_submission_content(date, submission_id)
            if content:
                # pre-score
                if self.sentiment_prescorer.predict(content):
                    contents.append(content)
                    logger.info(f"Adding {submission_id} to contents")
                else:
                    skipped_count += 1
                    logger.info(f"Skipping {submission_id} because it is not relevant")
        logger.info(f"Skipped {skipped_count}/{len(submission_ids_to_process)} submissions")
        if not contents:
            logger.warning(f"No content loaded for {date}")
            return DailyUpdate(
                date=date,
                submission_results={},
                is_complete_day=True,
                update_timestamp=datetime.now(),
                metadata={'warning': 'no_content_loaded'}
            )
        
        # Process the batch
        batch_result = await self.sentiment_processor.process_batch(contents)
        
        # Create daily update
        update = DailyUpdate(
            date=date,
            submission_results=batch_result.results,
            is_complete_day=not force_reprocess,
            update_timestamp=datetime.now(),
            metadata={
                'total_submissions_in_date': len(all_submission_ids),
                'processed_submissions': batch_result.successful_submissions,
                'failed_submissions': batch_result.failed_submissions,
                'total_extractions': batch_result.total_extractions,
                'processing_time': batch_result.processing_time
            }
        )
        
        # Save the update
        self.sentiment_storage.save_daily_update(update)
        
        logger.info(f"✅ Completed processing for {date}: "
                   f"{batch_result.successful_submissions} successful, "
                   f"{batch_result.failed_submissions} failed")
        
        return update
    
    async def update_day(self, date: str) -> DailyUpdate:
        """Check for and process any new submissions for a day"""
        logger.info(f"Checking for new submissions to process for {date}")
        
        # Get all submissions currently available for this date
        all_current_submission_ids = self.content_loader.get_submissions_for_date(date)
        
        if not all_current_submission_ids:
            logger.info(f"No submissions found for {date}")
            return DailyUpdate(
                date=date,
                submission_results={},
                is_complete_day=True,
                update_timestamp=datetime.now(),
                metadata={
                    'update_type': 'check_for_new',
                    'result': 'no_submissions_found'
                }
            )
        
        # Get submissions that haven't been processed yet
        unprocessed_submission_ids = self._get_unprocessed_submissions_for_date(date, all_current_submission_ids)
        
        if not unprocessed_submission_ids:
            logger.info(f"No new submissions to process for {date}. "
                       f"All {len(all_current_submission_ids)} submissions already processed.")
            return DailyUpdate(
                date=date,
                submission_results={},
                is_complete_day=True,
                update_timestamp=datetime.now(),
                metadata={
                    'update_type': 'check_for_new',
                    'result': 'no_new_submissions',
                    'total_submissions_available': len(all_current_submission_ids),
                    'already_processed': len(all_current_submission_ids)
                }
            )
        
        logger.info(f"Found {len(unprocessed_submission_ids)} new submissions to process for {date}")
        
        # Load content for new submissions
        contents = []
        failed_to_load = []
        
        for submission_id in unprocessed_submission_ids:
            content = self.content_loader.load_submission_content(date, submission_id)
            if content:
                contents.append(content)
            else:
                failed_to_load.append(submission_id)
        
        if not contents:
            logger.warning(f"No content could be loaded for new submissions in {date}")
            return DailyUpdate(
                date=date,
                submission_results={},
                is_complete_day=False,
                update_timestamp=datetime.now(),
                metadata={
                    'update_type': 'check_for_new',
                    'result': 'no_content_loaded',
                    'new_submissions_found': len(unprocessed_submission_ids),
                    'failed_to_load': len(failed_to_load)
                }
            )
        
        logger.info(f"Processing {len(contents)} new submissions for {date}")
        
        # Process the new submissions
        batch_result = await self.sentiment_processor.process_batch(contents)
        
        # Create update
        update = DailyUpdate(
            date=date,
            submission_results=batch_result.results,
            is_complete_day=False,
            update_timestamp=datetime.now(),
            metadata={
                'update_type': 'incremental_new_submissions',
                'total_submissions_available': len(all_current_submission_ids),
                'new_submissions_found': len(unprocessed_submission_ids),
                'new_submissions_loaded': len(contents),
                'new_submissions_processed': batch_result.successful_submissions,
                'new_submissions_failed': batch_result.failed_submissions,
                'failed_to_load': len(failed_to_load),
                'total_extractions': batch_result.total_extractions,
                'processing_time': batch_result.processing_time
            }
        )
        
        # Save the update
        self.sentiment_storage.save_daily_update(update)
        
        logger.info(f"✅ Updated {date}: processed {batch_result.successful_submissions} new submissions, "
                   f"{batch_result.failed_submissions} failed")
        
        return update


class IncrementalSentimentOrchestrator(BaseSentimentOrchestrator):
    """Orchestrator with incremental processing and checkpoint support"""
    
    def __init__(self, 
                 content_loader: ContentLoader,
                 sentiment_processor: SentimentProcessor,
                 sentiment_storage: IncrementalSentimentStorage,
                 sentiment_prescorer: SentimentPrescorer,
                 checkpoint_interval: int = 10,
                 orchestrator_batch_size: int = 10):
        """
        Initialize incremental orchestrator.
        
        Args:
            content_loader: Content loader for Reddit data
            sentiment_processor: Sentiment processor for extractions
            incremental_storage: Incremental storage for checkpoints and results
            checkpoint_interval: Save checkpoint every N submissions
            orchestrator_batch_size: Number of submissions to process in one batch
        """
        # Note: We pass incremental_storage as sentiment_storage to base class
        super().__init__(content_loader, sentiment_processor, sentiment_storage, sentiment_prescorer)
        self.incremental_storage = sentiment_storage  # Keep specific reference
        self.checkpoint_interval = checkpoint_interval
        self.sentiment_prescorer = sentiment_prescorer
        self.orchestrator_batch_size = orchestrator_batch_size 
    
    def _get_unprocessed_submissions_for_date(self, date: str, all_submission_ids: List[str]) -> List[str]:
        """Get unprocessed submissions using incremental storage"""
        return self.incremental_storage.get_unprocessed_submissions(date, all_submission_ids)
    
    async def process_day(self, date: str, 
                         force_reprocess_failed: bool = False,
                         session_id: Optional[str] = None) -> DailyUpdate:
        """
        Process day with incremental checkpointing.
        
        Args:
            date: Date to process (YYYYMMDD format)
            force_reprocess_failed: If True, retry previously failed submissions
            session_id: Optional session ID for tracking (auto-generated if None)
            
        Returns:
            DailyUpdate with processing results
        """
        if not session_id:
            session_id = f"{date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        logger.info(f"Starting incremental processing for {date} (session: {session_id})")
        
        # Get all available submissions
        all_submission_ids = self.content_loader.get_submissions_for_date(date)
        if not all_submission_ids:
            logger.warning(f"No submissions found for {date}")
            return self._create_empty_update(date, 'no_submissions_found')
        
        # Load or create checkpoint
        checkpoint = self.incremental_storage.load_checkpoint(date)
        if not checkpoint:
            checkpoint = ProcessingCheckpoint(
                date=date,
                processed_submissions=set(),
                failed_submissions=set(),
                last_checkpoint_time=datetime.now(),
                processing_session_id=session_id,
                total_submissions_found=len(all_submission_ids),
                metadata={}
            )
            logger.info(f"Created new checkpoint for {date}")
        else:
            logger.info(f"Resuming from checkpoint: {len(checkpoint.processed_submissions)} processed, "
                       f"{len(checkpoint.failed_submissions)} failed")
            checkpoint.processing_session_id = session_id  # Update session ID
            
            # Update total submissions found in case new ones were added
            checkpoint.total_submissions_found = len(all_submission_ids)
        
        # Determine what to process
        to_process = set(all_submission_ids) - checkpoint.processed_submissions
        
        if force_reprocess_failed:
            # Add failed submissions back to processing queue
            to_process.update(checkpoint.failed_submissions)
            checkpoint.failed_submissions.clear()
            logger.info(f"Reprocessing {len(checkpoint.failed_submissions)} previously failed submissions")
        else:
            # Exclude failed submissions from processing
            to_process -= checkpoint.failed_submissions
        
        if not to_process:
            logger.info(f"All submissions for {date} already processed")
            return self._create_empty_update(date, 'already_complete', checkpoint)
        logger.info(f"Processing {len(to_process)} submissions for {date}")
        
        # Load content for submissions to process
        contents_to_process = []
        failed_to_load = []
        skipped_to_process = []
        
        for submission_id in to_process:
            content = self.content_loader.load_submission_content(date, submission_id)
            if content:
                if self.sentiment_prescorer.predict(content):
                    contents_to_process.append(content)
                else:
                    skipped_to_process.append(submission_id)
                    logger.info(f"Skipping {submission_id} because it is not relevant")
            else:
                logger.warning(f"Could not load content for {submission_id}")
                checkpoint.failed_submissions.add(submission_id)
                failed_to_load.append(submission_id)
        
        if failed_to_load:
            logger.warning(f"Failed to load {len(failed_to_load)} submissions")

        if skipped_to_process:
            logger.warning(f"Skipped {len(skipped_to_process)} submissions")
        
        if not contents_to_process:
            logger.warning(f"No content could be loaded for processing {date}")
            # Save checkpoint with failed loads
            checkpoint.last_checkpoint_time = datetime.now()
            self.incremental_storage.save_checkpoint(checkpoint)
            return self._create_empty_update(date, 'no_content_loaded', checkpoint)
        
        # Process submissions one by one with checkpointing
        session_newly_successful_count = 0
        session_newly_failed_count = 0
        session_total_extractions = 0
        submissions_processed_this_checkpoint_period = 0

        processing_start_time = datetime.now()
        
        logger.info(f"Starting to process {len(contents_to_process)} submissions "
                f"in orchestrator batches of up to {self.orchestrator_batch_size}")
        

        for i in range(0, len(contents_to_process), self.orchestrator_batch_size):

            current_orchestrator_batch = contents_to_process[i:i + self.orchestrator_batch_size]
            
            if not current_orchestrator_batch:
                continue

            batch_num_display = (i // self.orchestrator_batch_size) + 1
            total_batches_display = (len(contents_to_process) + self.orchestrator_batch_size - 1) // self.orchestrator_batch_size
            
            logger.info(f"Processing orchestrator batch {batch_num_display}/{total_batches_display} "
                        f"with {len(current_orchestrator_batch)} submissions.")
            
            try:
                # Now, process_batch receives a list of content objects
                # sentiment_processor.process_batch itself will use extractor.extract_batch_async
                # which has its own internal batching for API calls.
                processor_batch_result = await self.sentiment_processor.process_batch(current_orchestrator_batch)

                # Iterate through results for each submission in the processed orchestrator batch
                for submission_id, result_detail in processor_batch_result.results.items():
                    if not result_detail.error:
                        self.incremental_storage.save_submission_result(date, submission_id, result_detail)
                        checkpoint.processed_submissions.add(submission_id)
                        if submission_id in checkpoint.failed_submissions: # If retrying a previously failed one
                            checkpoint.failed_submissions.remove(submission_id)
                        
                        session_newly_successful_count += 1
                        session_total_extractions += result_detail.extraction_count
                        logger.debug(f"✅ Processed {submission_id} from batch: {result_detail.extraction_count} extractions")
                    else:
                        checkpoint.failed_submissions.add(submission_id)
                        # If it was marked processed due to some edge case and now fails, ensure it's not in processed
                        if submission_id in checkpoint.processed_submissions:
                            checkpoint.processed_submissions.remove(submission_id)
                        session_newly_failed_count += 1
                        logger.warning(f"❌ Failed {submission_id} in batch (processor-reported): {result_detail.error}")
                    
                    submissions_processed_this_checkpoint_period += 1
                    # Save checkpoint based on individual submissions processed, even if mid-orchestrator-batch
                    if submissions_processed_this_checkpoint_period >= self.checkpoint_interval:
                        checkpoint.last_checkpoint_time = datetime.now()
                        self.incremental_storage.save_checkpoint(checkpoint)
                        logger.info(f"Checkpoint saved (mid-session): "
                                    f"{len(checkpoint.processed_submissions)} processed, "
                                    f"{len(checkpoint.failed_submissions)} failed. "
                                    f"{submissions_processed_this_checkpoint_period} submissions processed this period.")
                        submissions_processed_this_checkpoint_period = 0 # Reset counter

            except Exception as e:
                # This catches a failure in the self.sentiment_processor.process_batch() call itself for the current_orchestrator_batch
                logger.error(f"❌ Exception during processing of orchestrator batch {batch_num_display}: {e}. "
                            f"Marking all {len(current_orchestrator_batch)} submissions in this specific batch as failed (if not already processed).")
                
                for content_item_in_failed_batch in current_orchestrator_batch:
                    sub_id = content_item_in_failed_batch.submission_id
                    # Only mark as failed if it's not already successfully processed from a *previous* session/batch
                    # and avoid double-counting if it was already in the failed set.
                    if sub_id not in checkpoint.processed_submissions:
                        if sub_id not in checkpoint.failed_submissions: # Add to failed and count as newly failed
                            checkpoint.failed_submissions.add(sub_id)
                            session_newly_failed_count += 1
                        # If already in failed_submissions, it's not "newly" failed by this exception,
                        # but ensure it remains there.
                    
                    submissions_processed_this_checkpoint_period += 1 # Count as attempted

                # Potentially save checkpoint after a batch-level failure
                if submissions_processed_this_checkpoint_period >= self.checkpoint_interval:
                    checkpoint.last_checkpoint_time = datetime.now()
                    self.incremental_storage.save_checkpoint(checkpoint)
                    logger.info(f"Checkpoint saved (after orchestrator batch exception): "
                                f"{len(checkpoint.processed_submissions)} processed, "
                                f"{len(checkpoint.failed_submissions)} failed. "
                                f"{submissions_processed_this_checkpoint_period} submissions handled this period.")
                    submissions_processed_this_checkpoint_period = 0

        
        # Final checkpoint save
        checkpoint.last_checkpoint_time = datetime.now()
        session_processing_time_seconds = (datetime.now() - processing_start_time).total_seconds()
        checkpoint.metadata.update({
            'last_session_id': session_id, # Assuming session_id is defined
            'last_processing_time': session_processing_time_seconds,
            'last_session_successful': session_newly_successful_count,
            'last_session_failed': session_newly_failed_count,
            'last_session_extractions': session_total_extractions
            # Add 'last_session_failed_to_load': failed_to_load_count if you track it here
        })
        self.incremental_storage.save_checkpoint(checkpoint)
        logger.info(f"Final checkpoint for session {session_id} saved.")
        
        processing_time = (datetime.now() - processing_start_time).total_seconds()
        
        # Create update summary
        update = DailyUpdate(
            date=date,
            submission_results={},  # Results are saved incrementally
            is_complete_day=len(checkpoint.processed_submissions) + len(checkpoint.failed_submissions) >= len(all_submission_ids),
            update_timestamp=datetime.now(),
            metadata={
                'session_id': session_id,
                'total_submissions_available': len(all_submission_ids),
                'submissions_to_process': len(to_process),
                'submissions_loaded': len(contents_to_process),
                'failed_to_load': len(failed_to_load),
                'newly_processed': session_newly_successful_count,
                'newly_failed': session_newly_failed_count,
                'total_processed': len(checkpoint.processed_submissions),
                'total_failed': len(checkpoint.failed_submissions),
                'total_extractions_this_session': session_total_extractions,
                'processing_time': processing_time,
                'processing_complete': len(to_process) == 0,
                'checkpoint_interval': self.checkpoint_interval,
                'force_reprocess_failed': force_reprocess_failed
            }
        )
        
        logger.info(f"✅ Incremental processing complete for {date}:")
        logger.info(f"  Session: {session_id}")
        logger.info(f"  Newly processed: {session_newly_successful_count}")
        logger.info(f"  Newly failed: {session_newly_failed_count}")
        logger.info(f"  Total processed: {len(checkpoint.processed_submissions)}")
        logger.info(f"  Total failed: {len(checkpoint.failed_submissions)}")
        logger.info(f"  Processing time: {processing_time:.1f}s")
        logger.info(f"  Complete: {update.is_complete_day}")
        
        return update
    
    async def update_day(self, date: str, session_id: Optional[str] = None) -> DailyUpdate:
        """
        Update day incrementally by processing only new submissions in batches.
        
        Args:
            date: Date to update (YYYYMMDD format)
            session_id: Optional session ID for tracking
            
        Returns:
            DailyUpdate with results of processing new submissions
        """
        if not session_id:
            session_id = f"update_{date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        logger.info(f"Checking for new submissions to process for {date} (session: {session_id})")
        
        # Get all submissions currently available for this date
        all_current_submission_ids = self.content_loader.get_submissions_for_date(date)
        
        if not all_current_submission_ids:
            logger.info(f"No submissions found for {date}")
            return self._create_empty_update(date, 'no_submissions_found')
        
        # Get submissions that haven't been processed yet
        # This list contains only IDs of submissions that are neither processed nor failed.
        unprocessed_submission_ids_list = self.incremental_storage.get_unprocessed_submissions(
            date, all_current_submission_ids
        )
        
        if not unprocessed_submission_ids_list:
            logger.info(f"No new submissions to process for {date}. "
                       f"All {len(all_current_submission_ids)} submissions already handled.")
            
            checkpoint = self.incremental_storage.load_checkpoint(date)
            return DailyUpdate(
                date=date,
                submission_results={},
                is_complete_day=True if checkpoint and (len(checkpoint.processed_submissions) + len(checkpoint.failed_submissions) >= len(all_current_submission_ids)) else False,
                update_timestamp=datetime.now(),
                metadata={
                    'session_id': session_id,
                    'update_type': 'check_for_new',
                    'result': 'no_new_submissions',
                    'total_submissions_available': len(all_current_submission_ids),
                    'total_processed': len(checkpoint.processed_submissions) if checkpoint else 0,
                    'total_failed': len(checkpoint.failed_submissions) if checkpoint else 0
                }
            )
        
        logger.info(f"Found {len(unprocessed_submission_ids_list)} new submissions to process for {date}")
        
        # Load or create checkpoint
        checkpoint = self.incremental_storage.load_checkpoint(date)
        if not checkpoint:
            checkpoint = ProcessingCheckpoint(
                date=date,
                processed_submissions=set(),
                failed_submissions=set(),
                last_checkpoint_time=datetime.now(),
                processing_session_id=session_id,
                total_submissions_found=len(all_current_submission_ids),
                metadata={}
            )
            logger.info(f"Created new checkpoint for {date} during update")
        else:
            checkpoint.processing_session_id = session_id # Update session ID for this run
            checkpoint.total_submissions_found = len(all_current_submission_ids) # Update total count
        
        # Load content for new submissions
        contents_to_process = []
        failed_to_load = []
        
        for submission_id in unprocessed_submission_ids_list:
            content = self.content_loader.load_submission_content(date, submission_id)
            if content:
                contents_to_process.append(content)
            else:
                logger.warning(f"Could not load content for new submission {submission_id}")
                # Add to failed_submissions in checkpoint if content cannot be loaded
                checkpoint.failed_submissions.add(submission_id) 
                failed_to_load.append(submission_id)
        
        if failed_to_load:
            logger.warning(f"Failed to load {len(failed_to_load)} new submissions. They have been marked as failed in the checkpoint.")
        
        if not contents_to_process:
            logger.warning(f"No content could be loaded for new submissions in {date} that were pending.")
            # Save checkpoint with any newly marked failed loads
            checkpoint.last_checkpoint_time = datetime.now()
            self.incremental_storage.save_checkpoint(checkpoint)
            
            return DailyUpdate(
                date=date,
                submission_results={},
                is_complete_day=False, # Day is not complete if we intended to process but couldn't load
                update_timestamp=datetime.now(),
                metadata={
                    'session_id': session_id,
                    'update_type': 'incremental_new_submissions',
                    'result': 'no_content_loaded_for_pending',
                    'new_submissions_found': len(unprocessed_submission_ids_list),
                    'failed_to_load': len(failed_to_load),
                    'total_submissions_available': len(all_current_submission_ids)
                }
            )
        
        logger.info(f"Processing {len(contents_to_process)} new submissions for {date} "
                    f"in orchestrator batches of up to {self.orchestrator_batch_size}")

        # Counters for this update session
        session_new_successful_count = 0
        session_new_failed_count = 0 # Failures during processing, not loading
        session_total_extractions = 0
        submissions_processed_this_checkpoint_period = 0
        processing_start_time = datetime.now()

        for i in range(0, len(contents_to_process), self.orchestrator_batch_size):
            current_orchestrator_batch = contents_to_process[i:i + self.orchestrator_batch_size]
            
            if not current_orchestrator_batch:
                continue

            batch_num_display = (i // self.orchestrator_batch_size) + 1
            total_batches_display = (len(contents_to_process) + self.orchestrator_batch_size - 1) // self.orchestrator_batch_size
            
            logger.info(f"Processing orchestrator batch {batch_num_display}/{total_batches_display} "
                        f"with {len(current_orchestrator_batch)} new submissions.")
            
            try:
                processor_batch_result = await self.sentiment_processor.process_batch(current_orchestrator_batch)

                for submission_id, result_detail in processor_batch_result.results.items():
                    if not result_detail.error:
                        self.incremental_storage.save_submission_result(date, submission_id, result_detail)
                        checkpoint.processed_submissions.add(submission_id)
                        # If it was somehow in failed (e.g. from a load failure in a previous update run), remove it
                        if submission_id in checkpoint.failed_submissions:
                            checkpoint.failed_submissions.remove(submission_id)
                        
                        session_new_successful_count += 1
                        session_total_extractions += result_detail.extraction_count
                        logger.debug(f"✅ Processed new submission {submission_id} from batch: {result_detail.extraction_count} extractions")
                    else:
                        checkpoint.failed_submissions.add(submission_id)
                        if submission_id in checkpoint.processed_submissions: # Should not happen for new items
                            checkpoint.processed_submissions.remove(submission_id)
                        session_new_failed_count += 1
                        logger.warning(f"❌ Failed new submission {submission_id} in batch (processor-reported): {result_detail.error}")
                    
                    submissions_processed_this_checkpoint_period += 1
                    if submissions_processed_this_checkpoint_period >= self.checkpoint_interval:
                        checkpoint.last_checkpoint_time = datetime.now()
                        self.incremental_storage.save_checkpoint(checkpoint)
                        logger.info(f"Update checkpoint saved (mid-session): "
                                    f"{len(checkpoint.processed_submissions)} total processed, "
                                    f"{len(checkpoint.failed_submissions)} total failed. "
                                    f"{submissions_processed_this_checkpoint_period} submissions handled this period.")
                        submissions_processed_this_checkpoint_period = 0

            except Exception as e:
                logger.error(f"❌ Exception during processing of orchestrator batch {batch_num_display} of new submissions: {e}. "
                            f"Marking all {len(current_orchestrator_batch)} submissions in this specific batch as failed.")
                
                for content_item_in_failed_batch in current_orchestrator_batch:
                    sub_id = content_item_in_failed_batch.submission_id
                    if sub_id not in checkpoint.processed_submissions: # Should be true for new items
                        if sub_id not in checkpoint.failed_submissions:
                            checkpoint.failed_submissions.add(sub_id)
                            session_new_failed_count += 1
                    submissions_processed_this_checkpoint_period += 1

                if submissions_processed_this_checkpoint_period >= self.checkpoint_interval:
                    checkpoint.last_checkpoint_time = datetime.now()
                    self.incremental_storage.save_checkpoint(checkpoint)
                    logger.info(f"Update checkpoint saved (after orchestrator batch exception): "
                                f"{len(checkpoint.processed_submissions)} total processed, "
                                f"{len(checkpoint.failed_submissions)} total failed. "
                                f"{submissions_processed_this_checkpoint_period} submissions handled this period.")
                    submissions_processed_this_checkpoint_period = 0
        
        # Final checkpoint save for this update session
        checkpoint.last_checkpoint_time = datetime.now()
        session_processing_time_seconds = (datetime.now() - processing_start_time).total_seconds()
        checkpoint.metadata.update({
            'last_update_session_id': session_id,
            'last_update_time': session_processing_time_seconds,
            'last_update_successful': session_new_successful_count,
            'last_update_failed': session_new_failed_count, # Processing failures this session
            'last_update_extractions': session_total_extractions
        })
        self.incremental_storage.save_checkpoint(checkpoint)
        logger.info(f"Final update checkpoint for session {session_id} saved.")
        
        # Create update summary
        update = DailyUpdate(
            date=date,
            submission_results={},  # Results saved incrementally
            # Day is complete if all available submissions are either processed or failed
            is_complete_day=(len(checkpoint.processed_submissions) + len(checkpoint.failed_submissions)) >= len(all_current_submission_ids),
            update_timestamp=datetime.now(),
            metadata={
                'session_id': session_id,
                'update_type': 'incremental_new_submissions',
                'total_submissions_available': len(all_current_submission_ids),
                'new_submissions_found_for_update': len(unprocessed_submission_ids_list), # Total that were candidates
                'new_submissions_loaded_for_update': len(contents_to_process),
                'new_submissions_processed_this_session': session_new_successful_count,
                'new_submissions_failed_this_session': session_new_failed_count, # Processing failures
                'new_submissions_failed_to_load_this_session': len(failed_to_load),
                'total_extractions_this_session': session_total_extractions,
                'processing_time_this_session': session_processing_time_seconds,
                'total_processed_after_update': len(checkpoint.processed_submissions),
                'total_failed_after_update': len(checkpoint.failed_submissions)
            }
        )
        
        logger.info(f"✅ Update for {date} complete (session: {session_id}):")
        logger.info(f"  New submissions processed: {session_new_successful_count}")
        logger.info(f"  New submissions failed processing: {session_new_failed_count}")
        logger.info(f"  New submissions failed to load: {len(failed_to_load)}")
        logger.info(f"  Total processed for day: {len(checkpoint.processed_submissions)}")
        logger.info(f"  Total failed for day: {len(checkpoint.failed_submissions)}")
        logger.info(f"  Processing time: {session_processing_time_seconds:.1f}s")
        logger.info(f"  Day is complete: {update.is_complete_day}")
        
        return update
    
    def _create_empty_update(self, date: str, reason: str, 
                           checkpoint: Optional[ProcessingCheckpoint] = None) -> DailyUpdate:
        """Create empty update with reason and optional checkpoint info"""
        metadata = {'skipped_reason': reason}
        
        if checkpoint:
            metadata.update({
                'total_processed': len(checkpoint.processed_submissions),
                'total_failed': len(checkpoint.failed_submissions),
                'last_checkpoint': checkpoint.last_checkpoint_time.isoformat(),
                'session_id': checkpoint.processing_session_id
            })
        
        return DailyUpdate(
            date=date,
            submission_results={},
            is_complete_day=True,
            update_timestamp=datetime.now(),
            metadata=metadata
        )
    
    def finalize_day(self, date: str, cleanup_incremental: bool = True, force_finalize: bool = False) -> Optional[DailySentimentBatch]:
        """
        Finalize processing for a day by consolidating results.
        
        Args:
            date: Date to finalize
            cleanup_incremental: If True, remove incremental files after consolidation
            force_finalize: If True, finalize even if the day is not complete
        Returns:
            DailySentimentBatch if successful, None otherwise
        """
        logger.info(f"Finalizing day {date} (cleanup_incremental={cleanup_incremental})")
        
        # Check if day is complete
        status = self.get_processing_status(date)
        print(status)
        if not status['processing_complete']:
            logger.warning(f"Day {date} is not complete: {status['unprocessed']} submissions unprocessed")
            # Continue anyway but log warning
        
        # Consolidate incremental results
        daily_batch = self.incremental_storage.consolidate_daily_results(date)
        
        if not daily_batch:
            logger.error(f"Failed to consolidate results for {date}")
            return None
        
        if cleanup_incremental:
            try:
                # Clean up incremental files
                self.incremental_storage.cleanup_incremental_files(date)
                
                # Remove checkpoint
                checkpoint_file = self.incremental_storage.checkpoints_path / f"{date}_checkpoint.json"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    logger.info(f"Removed checkpoint for {date}")
                
                logger.info(f"Cleaned up incremental files for {date}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup incremental files for {date}: {e}")
                # Don't fail the finalization due to cleanup issues
        
        logger.info(f"✅ Finalized {date}: {daily_batch.total_submissions} submissions, "
                   f"{daily_batch.total_extractions} extractions")
        
        return daily_batch
    
    def get_processing_status(self, date: str) -> Dict[str, Any]:
        """
        Get comprehensive processing status for a date.
        
        Args:
            date: Date to check
            
        Returns:
            Dictionary with detailed processing status
        """
        all_submission_ids = self.content_loader.get_submissions_for_date(date)
        return self.incremental_storage.get_processing_summary(date, all_submission_ids)
    
    def retry_failed_submissions(self, date: str, 
                               specific_submission_ids: Optional[List[str]] = None,
                               session_id: Optional[str] = None) -> DailyUpdate:
        """
        Retry processing failed submissions for a date.
        
        Args:
            date: Date to retry failed submissions for
            specific_submission_ids: If provided, only retry these specific submissions
            session_id: Optional session ID for tracking
            
        Returns:
            DailyUpdate with retry results
        """
        if not session_id:
            session_id = f"retry_{date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        logger.info(f"Retrying failed submissions for {date} (session: {session_id})")
        
        # Load checkpoint to get failed submissions
        checkpoint = self.incremental_storage.load_checkpoint(date)
        if not checkpoint:
            logger.warning(f"No checkpoint found for {date}")
            return self._create_empty_update(date, 'no_checkpoint_found')
        
        failed_submissions = checkpoint.failed_submissions.copy()
        
        if specific_submission_ids:
            # Only retry specific submissions
            failed_submissions = failed_submissions.intersection(set(specific_submission_ids))
            logger.info(f"Retrying {len(failed_submissions)} specific failed submissions")
        
        if not failed_submissions:
            logger.info(f"No failed submissions to retry for {date}")
            return self._create_empty_update(date, 'no_failed_submissions', checkpoint)
        
        logger.info(f"Retrying {len(failed_submissions)} failed submissions for {date}")
        
        # Remove from failed set (will be re-added if they fail again)
        checkpoint.failed_submissions -= failed_submissions
        checkpoint.processing_session_id = session_id
        
        # This is essentially the same as processing, but only for specific submissions
        # We can reuse the process_day logic by temporarily modifying the checkpoint
        original_processed = checkpoint.processed_submissions.copy()
        
        # Temporarily remove the submissions we want to retry from processed set
        # (they shouldn't be in processed set anyway if they're in failed set)
        
        # Save the modified checkpoint
        self.incremental_storage.save_checkpoint(checkpoint)
        
        # Now call process_day which will process the unprocessed submissions
        # (which now includes our previously failed ones)
        return asyncio.run(self.process_day(date, force_reprocess_failed=False, session_id=session_id))
    
    def get_checkpoint_info(self, date: str) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint information for a date.
        
        Args:
            date: Date to get checkpoint info for
            
        Returns:
            Dictionary with checkpoint information or None if no checkpoint exists
        """
        checkpoint = self.incremental_storage.load_checkpoint(date)
        if not checkpoint:
            return None
        
        return {
            'date': checkpoint.date,
            'processed_count': len(checkpoint.processed_submissions),
            'failed_count': len(checkpoint.failed_submissions),
            'total_found': checkpoint.total_submissions_found,
            'last_checkpoint_time': checkpoint.last_checkpoint_time.isoformat(),
            'processing_session_id': checkpoint.processing_session_id,
            'metadata': checkpoint.metadata,
            'completion_rate': len(checkpoint.processed_submissions) / checkpoint.total_submissions_found if checkpoint.total_submissions_found > 0 else 0,
            'success_rate': len(checkpoint.processed_submissions) / (len(checkpoint.processed_submissions) + len(checkpoint.failed_submissions)) if (checkpoint.processed_submissions or checkpoint.failed_submissions) else 0
        }
    
    def list_processing_sessions(self, date: str) -> List[Dict[str, Any]]:
        """
        List all processing sessions for a date based on checkpoint metadata.
        
        Args:
            date: Date to list sessions for
            
        Returns:
            List of session information dictionaries
        """
        checkpoint = self.incremental_storage.load_checkpoint(date)
        if not checkpoint:
            return []
        
        sessions = []
        
        # Current/last session
        sessions.append({
            'session_id': checkpoint.processing_session_id,
            'session_type': 'current',
            'last_activity': checkpoint.last_checkpoint_time.isoformat(),
            'processed_count': len(checkpoint.processed_submissions),
            'failed_count': len(checkpoint.failed_submissions)
        })
        
        # Extract historical sessions from metadata if available
        metadata = checkpoint.metadata or {}
        
        if 'last_session_id' in metadata:
            sessions.append({
                'session_id': metadata['last_session_id'],
                'session_type': 'previous_process',
                'processing_time': metadata.get('last_processing_time', 0),
                'successful': metadata.get('last_session_successful', 0),
                'failed': metadata.get('last_session_failed', 0)
            })
        
        if 'last_update_session_id' in metadata:
            sessions.append({
                'session_id': metadata['last_update_session_id'],
                'session_type': 'previous_update',
                'processing_time': metadata.get('last_update_time', 0),
                'successful': metadata.get('last_update_successful', 0),
                'failed': metadata.get('last_update_failed', 0)
            })
        
        return sessions
    