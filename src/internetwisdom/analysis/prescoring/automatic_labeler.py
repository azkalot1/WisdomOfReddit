from typing import Optional
from .state import RelevanceResults
from .prompts import prescorer_system_prompt
from langchain_openai import ChatOpenAI
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ScoringResult:
    """Result from automatic scoring"""
    submission_path: str
    submission_id: str
    is_relevant: bool
    reasoning: str
    confidence: int
    text_length: int
    truncated: bool
    processing_time: float
    timestamp: datetime
    error: Optional[str] = None

@dataclass
class BatchScoringResults:
    """Results from batch scoring"""
    total_submissions: int
    successful_scores: int
    failed_scores: int
    relevant_count: int
    irrelevant_count: int
    avg_confidence: float
    total_processing_time: float
    results: List[ScoringResult]
    timestamp: datetime

class AutomaticRelevanceScorer:
    """
    Automatic relevance scoring using GPT-4 for Reddit submissions
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o",
                 temperature: float = 0,
                 max_tokens: Optional[int] = None,
                 timeout: int = 30,
                 max_retries: int = 2,
                 max_text_length: int = 8000,  # Reasonable limit for GPT-4
                 rate_limit_delay: float = 1.0):  # Delay between API calls
        """
        Initialize automatic scorer
        
        Args:
            model_name: OpenAI model to use
            temperature: Model temperature
            max_tokens: Max tokens for response
            timeout: Request timeout in seconds
            max_retries: Max retries for failed requests
            max_text_length: Max characters to send to model (will truncate)
            rate_limit_delay: Delay between API calls to avoid rate limits
        """
        self.model = ChatOpenAI(
            model=model_name,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries
        ).with_structured_output(RelevanceResults)
        
        self.max_text_length = max_text_length
        self.rate_limit_delay = rate_limit_delay
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_submission(self, submission_path: Path) -> Tuple[Dict[str, Any], str]:
        """
        Load submission from JSON file
        
        Args:
            submission_path: Path to submission JSON file
            
        Returns:
            Tuple of (submission_data, submission_id)
        """
        with open(submission_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract submission ID from filename or data
        submission_id = submission_path.stem
        if 'submission_id' in data:
            submission_id = data['submission_id']
        
        return data, submission_id
    
    def _create_text_from_submission(self, submission_data: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Create text string from submission data with truncation
        
        Args:
            submission_data: Submission JSON data
            
        Returns:
            Tuple of (text_string, was_truncated)
        """
        # Create text from all values in the submission
        text_parts = []
        
        # Prioritize important fields first
        priority_fields = ['title', 'selftext', 'body']
        other_fields = []
        
        for key, value in submission_data.items():
            if key in priority_fields:
                text_parts.append(f"{key}: {str(value)}")
            else:
                other_fields.append(f"{key}: {str(value)}")
        
        # Add other fields
        text_parts.extend(other_fields)
        
        # Join all parts
        full_text = '\n'.join(text_parts)
        
        # Truncate if necessary
        truncated = False
        if len(full_text) > self.max_text_length:
            truncated = True
            full_text = full_text[:self.max_text_length]
            # Try to truncate at a reasonable boundary
            last_newline = full_text.rfind('\n')
            if last_newline > self.max_text_length * 0.8:  # If we can save 20%+ by truncating at newline
                full_text = full_text[:last_newline]
            full_text += "\n[TEXT TRUNCATED]"
        
        return full_text, truncated
    

    def score_single_submission(self, submission_path: Path) -> ScoringResult:
        """
        Score a single submission for relevance
        
        Args:
            submission_path: Path to submission JSON file
            
        Returns:
            ScoringResult with scoring information
        """
        start_time = time.time()
        submission_path = Path(submission_path)
        
        try:
            # Load submission
            submission_data, submission_id = self._load_submission(submission_path)
            
            # Create text
            text, truncated = self._create_text_from_submission(submission_data)
            
            # Create messages
            messages = [
                SystemMessage(content=prescorer_system_prompt),
                HumanMessage(content=text)
            ]
            
            # Get response from model
            self.logger.info(f"Scoring submission {submission_id}...")
            response = self.model.invoke(messages)
            
            processing_time = time.time() - start_time
            
            return ScoringResult(
                submission_path=str(submission_path),
                submission_id=submission_id,
                is_relevant=response.is_relevant,
                reasoning=response.reasoning,
                confidence=response.confidence,
                text_length=len(text),
                truncated=truncated,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error scoring {submission_path}: {e}")
            
            return ScoringResult(
                submission_path=str(submission_path),
                submission_id=submission_path.stem,
                is_relevant=False,  # Default to irrelevant on error
                reasoning=f"Error during processing: {str(e)}",
                confidence=0,
                text_length=0,
                truncated=False,
                processing_time=processing_time,
                timestamp=datetime.now(),
                error=str(e)
            )
        

    def score_batch(self, submission_paths: List[Path], 
                   show_progress: bool = True) -> BatchScoringResults:
        """
        Score multiple submissions for relevance
        
        Args:
            submission_paths: List of paths to submission JSON files
            show_progress: Whether to show progress updates
            
        Returns:
            BatchScoringResults with batch scoring information
        """
        start_time = time.time()
        results = []
        
        self.logger.info(f"Starting batch scoring of {len(submission_paths)} submissions...")
        
        for i, path in enumerate(submission_paths):
            # Score submission
            result = self.score_single_submission(path)
            results.append(result)
            
            # Rate limiting
            if i < len(submission_paths) - 1:  # Don't delay after last item
                time.sleep(self.rate_limit_delay)
            
            # Progress updates
            if show_progress and (i + 1) % 10 == 0:
                relevant_so_far = sum(1 for r in results if r.is_relevant and not r.error)
                self.logger.info(f"Progress: {i + 1}/{len(submission_paths)} "
                               f"({relevant_so_far} relevant so far)")
        
        # Calculate summary statistics
        successful_results = [r for r in results if not r.error]
        failed_results = [r for r in results if r.error]
        relevant_results = [r for r in successful_results if r.is_relevant]
        
        total_processing_time = time.time() - start_time
        avg_confidence = (sum(r.confidence for r in successful_results) / 
                         len(successful_results) if successful_results else 0)
        
        batch_results = BatchScoringResults(
            total_submissions=len(submission_paths),
            successful_scores=len(successful_results),
            failed_scores=len(failed_results),
            relevant_count=len(relevant_results),
            irrelevant_count=len(successful_results) - len(relevant_results),
            avg_confidence=avg_confidence,
            total_processing_time=total_processing_time,
            results=results,
            timestamp=datetime.now()
        )
        
        self.logger.info(f"Batch scoring complete!")
        self.logger.info(f"  Total: {batch_results.total_submissions}")
        self.logger.info(f"  Successful: {batch_results.successful_scores}")
        self.logger.info(f"  Failed: {batch_results.failed_scores}")
        self.logger.info(f"  Relevant: {batch_results.relevant_count}")
        self.logger.info(f"  Irrelevant: {batch_results.irrelevant_count}")
        self.logger.info(f"  Avg Confidence: {batch_results.avg_confidence:.1f}")
        self.logger.info(f"  Total Time: {batch_results.total_processing_time:.1f}s")
        
        return batch_results