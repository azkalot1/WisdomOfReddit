from typing import Protocol, Dict, Any, List, Optional
from .state import RedditContent, BatchProcessingResult, ProcessingResult
from .async_extraction import AsyncSentimentExtractor, AsyncExtractionConfig
from .state import ModelConfig
from datetime import datetime

class SentimentProcessor(Protocol):
    """Interface for processing sentiments"""
    
    async def process_batch(self, contents: List[RedditContent]) -> BatchProcessingResult:
        """Process a batch of Reddit content and return results"""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        ...

class AsyncSentimentProcessor:
    """Async sentiment processor implementation"""
    
    def __init__(self, 
                 model_config: Optional[ModelConfig] = None,
                 async_config: Optional[AsyncExtractionConfig] = None):
        self.extractor = AsyncSentimentExtractor(
            model_config=model_config,
            async_config=async_config
        )
    
    async def process_batch(self, contents: List[RedditContent]) -> BatchProcessingResult:
        """Process a batch of Reddit content and return results"""
        start_time = datetime.now()
        
        # Extract sentiments using the async extractor
        extraction_results = await self.extractor.extract_batch_async(
            contents=contents,
            show_progress=True
        )
        
        # Convert to ProcessingResult format
        processing_results = {}
        successful_count = 0
        failed_count = 0
        total_extractions = 0
        
        for content in contents:
            submission_id = content.submission_id or f"unknown_{id(content)}"
            
            if submission_id in extraction_results and extraction_results[submission_id]:
                # Successful processing
                sentiments = extraction_results[submission_id]
                processing_results[submission_id] = ProcessingResult(
                    submission_id=submission_id,
                    sentiments=sentiments,
                    processing_time=0.0,  # Individual timing not tracked in batch
                    extraction_count=len(sentiments),
                    error=None,
                    metadata={'subreddit': content.subreddit}
                )
                successful_count += 1
                total_extractions += len(sentiments)
            else:
                # Failed processing
                processing_results[submission_id] = ProcessingResult(
                    submission_id=submission_id,
                    sentiments=[],
                    processing_time=0.0,
                    extraction_count=0,
                    error="Processing failed or no results",
                    metadata={'subreddit': content.subreddit}
                )
                failed_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchProcessingResult(
            results=processing_results,
            total_submissions=len(contents),
            successful_submissions=successful_count,
            failed_submissions=failed_count,
            total_extractions=total_extractions,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return self.extractor.get_metrics()