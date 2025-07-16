import asyncio
import aiohttp
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, AsyncRetrying
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import tiktoken
from collections import deque
from dataclasses import dataclass
import time
from pydantic import BaseModel

from .state import AsyncExtractionConfig, ModelConfig, SentimentResults, RedditContent, ExtractionError
from .prompts import extractor_system_prompt, extractor_content

logger = logging.getLogger(__name__)


class AsyncSentimentExtractor:
    """
    Async version of SentimentExtractor with parallel processing support.
    
    Efficiently utilizes API rate limits by processing multiple extractions
    concurrently while respecting rate limits.
    """
    
    def __init__(
        self, 
        model_config: Optional[ModelConfig] = None,
        async_config: Optional[AsyncExtractionConfig] = None,
        system_prompt: Optional[str] = None,
        content_template: Optional[str] = None,
        output_model: Optional[BaseModel] = None
    ):
        """
        Initialize the async sentiment extractor.
        
        Args:
            model_config: Configuration for the LLM model
            async_config: Configuration for async processing
            system_prompt: System prompt for extraction (uses default if None)
            content_template: Template for formatting content (uses default if None)
        """
        self.model_config = model_config or ModelConfig()
        self.async_config = async_config or AsyncExtractionConfig()
        self.system_prompt = system_prompt or extractor_system_prompt
        self.content_template = content_template or extractor_content
        self.output_model = output_model or SentimentResults
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize tokenizer
        self.encoding = self._get_encoding()
        
        # Rate limiting
        self._rate_limiter = AsyncRateLimiter(
            requests_per_minute=self.async_config.requests_per_minute,
            tokens_per_minute=self.async_config.tokens_per_minute,
            token_buffer_ratio=self.async_config.token_buffer_ratio,
            request_buffer_ratio=self.async_config.request_buffer_ratio,
            enabled=self.async_config.enable_rate_limiting
        )
        
        # Metrics
        self.extraction_count = 0
        self.error_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._start_time = None
        
        logger.info(f"Initialized AsyncSentimentExtractor with model: {self.model_config.model_name}")
        logger.info(f"Async config: {self.async_config.max_concurrent_requests} concurrent, "
                   f"{self.async_config.requests_per_minute} RPM")
    
    def _initialize_model(self) -> ChatOpenAI:
        """Initialize the LLM with structured output support."""
        try:
            model = ChatOpenAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                timeout=self.model_config.timeout,
                max_retries=self.model_config.max_retries
            )
            return model.with_structured_output(SentimentResults)
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise ExtractionError(f"Model initialization failed: {e}")
    
    async def extract_all_async(self, content: RedditContent) -> List[SentimentResults]:
        """
        Extract sentiment from submission and each comment individually (async).
        
        Args:
            content: RedditContent object containing submission and comments
            
        Returns:
            List of SentimentResults, one for each extraction
        """
        extraction_tasks = []
        
        # If no comments, process submission alone
        if not content.comments:
            logger.debug(f"Processing submission alone (no comments) for {content.submission_id}")
            task = self._extract_single_async(
                title=content.title,
                submission_body=content.submission_body,
                comment_text="",
                extraction_context="submission_only"
            )
            extraction_tasks.append(task)
        else:
            # Create tasks for submission + each comment
            for i, comment in enumerate(content.comments):
                logger.debug(f"Creating task for submission + comment {i+1}/{len(content.comments)}")
                task = self._extract_single_async(
                    title=content.title,
                    submission_body=content.submission_body,
                    comment_text=comment,
                    extraction_context=f"comment_{i+1}"
                )
                extraction_tasks.append(task)
        
        # Execute all extractions concurrently
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Extraction {i+1} failed: {result}")
                self.error_count += 1
            else:
                successful_results.append(result)
                self.extraction_count += 1
        
        logger.info(
            f"Completed async extraction for {content.submission_id}: "
            f"{len(successful_results)}/{len(extraction_tasks)} extractions successful"
        )
        
        return successful_results
    
    async def _extract_single_async(self, 
                                    title: str, 
                                    submission_body: str, 
                                    comment_text: str,
                                    extraction_context: str = "") -> SentimentResults:
        """Extract sentiment with token-aware rate limiting"""
        
        # Format content and estimate tokens
        formatted_content = self._format_single_extraction(title, submission_body, comment_text)
        
        # Estimate tokens for this request
        system_tokens = len(self.encoding.encode(self.system_prompt))
        content_tokens = len(self.encoding.encode(formatted_content))
        estimated_input_tokens = system_tokens + content_tokens
        estimated_output_tokens = 300  # Rough estimate
        total_estimated_tokens = estimated_input_tokens + estimated_output_tokens
        
        # Rate limiting with token awareness
        rate_limit_info = await self._rate_limiter.acquire(total_estimated_tokens)
        
        if rate_limit_info['delay'] > 0:
            logger.debug(f"Rate limited for {rate_limit_info['delay']:.2f}s due to {rate_limit_info['reason']}")
        
        # Retry logic with async
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True
        ):
            with attempt:
                try:
                    # Create messages for LLM
                    messages = [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=formatted_content)
                    ]
                    
                    # Extract sentiment (async)
                    start_time = datetime.now()
                    response = await self.model.ainvoke(messages)
                    extraction_time = (datetime.now() - start_time).total_seconds()
                    
                    # Update actual token usage (if available from response)
                    # Note: You might need to modify this based on how your LangChain setup provides token info
                    actual_input_tokens = estimated_input_tokens  # Use estimate if actual not available
                    actual_output_tokens = estimated_output_tokens  # Use estimate if actual not available
                    
                    # Try to get actual token usage from response metadata if available
                    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                        token_usage = response.response_metadata['token_usage']
                        actual_input_tokens = token_usage.get('prompt_tokens', estimated_input_tokens)
                        actual_output_tokens = token_usage.get('completion_tokens', estimated_output_tokens)
                    
                    # Update rate limiter with actual usage
                    self._rate_limiter.update_actual_usage(actual_input_tokens, actual_output_tokens)
                    
                    # Update metrics
                    self.total_input_tokens += actual_input_tokens
                    self.total_output_tokens += actual_output_tokens
                    
                    # Validate response
                    if not isinstance(response, SentimentResults):
                        raise ExtractionError(f"Invalid response type: {type(response)}")
                    
                    # Post-process results
                    response = self._post_process_results(response)
                    
                    # Add extraction context metadata
                    if hasattr(response, 'extraction_context'):
                        response.extraction_context = extraction_context
                    
                    logger.debug(
                        f"Async extraction {extraction_context} completed in {extraction_time:.2f}s. "
                        f"Tokens: {actual_input_tokens}+{actual_output_tokens}={actual_input_tokens + actual_output_tokens}. "
                        f"Found {len(response.sentiments)} sentiments."
                    )
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Async extraction failed for {extraction_context}: {e}")
                    raise ExtractionError(f"Async extraction failed: {e}") from e
    
    async def extract_batch_async(
        self, 
        contents: List[RedditContent], 
        continue_on_error: bool = True,
        show_progress: bool = True
    ) -> Dict[str, List[SentimentResults]]:
        """
        Extract sentiment from multiple Reddit contents concurrently.
        
        Args:
            contents: List of RedditContent objects
            continue_on_error: Whether to continue processing on errors
            show_progress: Whether to show progress updates
            
        Returns:
            Dictionary mapping submission_id to list of SentimentResults
        """
        self._start_time = time.time()
        all_results = {}
        
        # Process in batches to manage memory and rate limits
        batch_size = self.async_config.batch_size
        total_batches = (len(contents) + batch_size - 1) // batch_size
        
        logger.info(f"Starting async batch extraction: {len(contents)} submissions in {total_batches} batches")
        
        for batch_idx in range(0, len(contents), batch_size):
            batch_contents = contents[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_contents)} submissions)")
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.async_config.max_concurrent_requests)
            
            # Create tasks for this batch
            batch_tasks = []
            for content in batch_contents:
                task = self._process_single_content_async(content, semaphore, continue_on_error)
                batch_tasks.append(task)
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for content, result in zip(batch_contents, batch_results):
                submission_id = content.submission_id or f"submission_{id(content)}"
                
                if isinstance(result, Exception):
                    logger.error(f"Failed to process submission {submission_id}: {result}")
                    if continue_on_error:
                        all_results[submission_id] = []
                    else:
                        raise result
                else:
                    all_results[submission_id] = result
            
            # Progress update
            if show_progress:
                processed_so_far = min((batch_num) * batch_size, len(contents))
                successful_so_far = sum(1 for results in all_results.values() if results)
                total_extractions_so_far = sum(len(results) for results in all_results.values())
                
                logger.info(
                    f"Progress: {processed_so_far}/{len(contents)} submissions, "
                    f"{successful_so_far} successful, {total_extractions_so_far} total extractions"
                )
            
            # Delay between batches if not the last batch
            if batch_num < total_batches and self.async_config.delay_between_batches > 0:
                await asyncio.sleep(self.async_config.delay_between_batches)
        
        # Final summary
        total_submissions = len(contents)
        successful_submissions = sum(1 for results in all_results.values() if results)
        total_extractions = sum(len(results) for results in all_results.values())
        total_time = time.time() - self._start_time
        
        logger.info(
            f"Async batch extraction complete: {successful_submissions}/{total_submissions} "
            f"submissions processed, {total_extractions} total extractions in {total_time:.1f}s"
        )
        logger.info(f"Average: {total_extractions/total_time:.1f} extractions/second")
        
        return all_results
    
    async def _process_single_content_async(
        self, 
        content: RedditContent, 
        semaphore: asyncio.Semaphore,
        continue_on_error: bool
    ) -> List[SentimentResults]:
        """Process a single content with semaphore control."""
        async with semaphore:
            try:
                return await self.extract_all_async(content)
            except Exception as e:
                if continue_on_error:
                    return []
                else:
                    raise
    
    def _format_single_extraction(
        self, 
        title: str, 
        submission_body: str, 
        comment_text: str
    ) -> str:
        """Format content for a single extraction."""
        if not comment_text:
            comments_section = "No comments"
        else:
            comments_section = comment_text
        
        formatted = self.content_template.format(
            title_text=title,
            submission_text=submission_body or "No submission body",
            comments_text=comments_section
        )
        
        return formatted
    
    def _post_process_results(self, results: SentimentResults) -> SentimentResults:
        """Post-process extraction results."""
        # Normalize tickers
        for sentiment in results.sentiments:
            sentiment.ticker = self._normalize_ticker(sentiment.ticker)
        
        if results.primary_ticker:
            results.primary_ticker = self._normalize_ticker(results.primary_ticker)
        
        return results
    
    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        """Normalize ticker symbol (remove $ prefix, uppercase)."""
        if not ticker:
            return ticker
        return ticker.replace('$', '').upper().strip()
    
    def _get_encoding(self):
        """Get the appropriate encoding for the model."""
        model_name = self.model_config.model_name
        
        if "gpt-4" in model_name:
            return tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model_name:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            return tiktoken.get_encoding("cl100k_base")
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting status"""
        return {
            'rate_limiter_stats': self._rate_limiter.get_stats(),
            'current_usage': self._rate_limiter.get_current_usage(),
            'can_make_request': self._rate_limiter.can_make_request()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get enhanced extraction metrics"""
        total_time = time.time() - self._start_time if self._start_time else 0
        
        base_metrics = {
            'total_extractions': self.extraction_count,
            'total_errors': self.error_count,
            'success_rate': (
                self.extraction_count / (self.extraction_count + self.error_count)
                if (self.extraction_count + self.error_count) > 0 
                else 0
            ),
            'total_processing_time': total_time,
            'extractions_per_second': self.extraction_count / total_time if total_time > 0 else 0,
            'model': self.model_config.model_name,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'avg_tokens_per_extraction': (
                (self.total_input_tokens + self.total_output_tokens) / self.extraction_count
                if self.extraction_count > 0 else 0
            )
        }
        
        # Add rate limiting metrics
        base_metrics.update(self.get_rate_limit_status())
        
        return base_metrics
    
    def estimate_tokens(self, content: RedditContent) -> Dict[str, Any]:
        """
        Estimate the number of tokens needed for processing a RedditContent.
        
        This calculates tokens for:
        - System prompt (once)
        - Each submission + comment pair
        
        Args:
            content: RedditContent object
            
        Returns:
            Dictionary with token estimates and cost projections
        """
        # Count system prompt tokens (used for each extraction)
        system_tokens = len(self.encoding.encode(self.system_prompt))
        
        # Estimate tokens for each extraction
        extraction_estimates = []
        
        # If no comments, estimate for submission alone
        if not content.comments:
            formatted = self._format_single_extraction(
                content.title,
                content.submission_body,
                ""
            )
            content_tokens = len(self.encoding.encode(formatted))
            total_tokens = system_tokens + content_tokens
            
            extraction_estimates.append({
                'extraction': 'submission_only',
                'content_tokens': content_tokens,
                'total_tokens': total_tokens
            })
        else:
            # Estimate for each comment
            for i, comment in enumerate(content.comments):
                formatted = self._format_single_extraction(
                    content.title,
                    content.submission_body,
                    comment
                )
                content_tokens = len(self.encoding.encode(formatted))
                total_tokens = system_tokens + content_tokens
                
                extraction_estimates.append({
                    'extraction': f'comment_{i+1}',
                    'content_tokens': content_tokens,
                    'total_tokens': total_tokens
                })
        
        # Calculate totals
        total_input_tokens = sum(est['total_tokens'] for est in extraction_estimates)
        num_extractions = len(extraction_estimates)
        
        # Estimate output tokens (rough estimate based on typical response size)
        estimated_output_tokens_per_extraction = 300  # Adjust based on your experience
        total_output_tokens = num_extractions * estimated_output_tokens_per_extraction
        
        # Calculate costs (example rates - adjust to your model's pricing)
        cost_per_1k_input = self._get_input_token_cost()
        cost_per_1k_output = self._get_output_token_cost()
        
        input_cost = (total_input_tokens /  1_000_000) * cost_per_1k_input
        output_cost = (total_output_tokens /  1_000_000) * cost_per_1k_output
        total_cost = input_cost + output_cost
        
        return {
            'submission_id': content.submission_id,
            'num_extractions': num_extractions,
            'system_prompt_tokens': system_tokens,
            'total_input_tokens': total_input_tokens,
            'estimated_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'extraction_details': extraction_estimates,
            'cost_estimate': {
                'input_cost': f"${input_cost:.4f}",
                'output_cost': f"${output_cost:.4f}",
                'total_cost': f"${total_cost:.4f}",
                'cost_per_extraction': f"${total_cost / num_extractions:.4f}"
            },
            'token_summary': {
                'avg_tokens_per_extraction': total_input_tokens / num_extractions,
                'max_tokens_single_extraction': max(est['total_tokens'] for est in extraction_estimates),
                'min_tokens_single_extraction': min(est['total_tokens'] for est in extraction_estimates)
            }
        }
    
    def estimate_batch_tokens(self, contents: List[RedditContent]) -> Dict[str, Any]:
        """
        Estimate tokens for a batch of RedditContent objects.
        
        Args:
            contents: List of RedditContent objects
            
        Returns:
            Dictionary with batch token estimates and cost projections
        """
        batch_estimates = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_extractions = 0
        
        for content in contents:
            estimate = self.estimate_tokens(content)
            batch_estimates.append(estimate)
            total_input_tokens += estimate['total_input_tokens']
            total_output_tokens += estimate['estimated_output_tokens']
            total_extractions += estimate['num_extractions']
        
        # Calculate costs
        cost_per_1k_input = self._get_input_token_cost()
        cost_per_1k_output = self._get_output_token_cost()
        
        input_cost = (total_input_tokens / 1_000_000) * cost_per_1k_input
        output_cost = (total_output_tokens /  1_000_000) * cost_per_1k_output
        total_cost = input_cost + output_cost
        
        return {
            'num_submissions': len(contents),
            'total_extractions': total_extractions,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'cost_estimate': {
                'input_cost': f"${input_cost:.2f}",
                'output_cost': f"${output_cost:.2f}",
                'total_cost': f"${total_cost:.2f}",
                'avg_cost_per_submission': f"${total_cost / len(contents):.4f}",
                'avg_cost_per_extraction': f"${total_cost / total_extractions:.4f}"
            },
            'token_summary': {
                'avg_input_tokens_per_submission': total_input_tokens / len(contents),
                'avg_input_tokens_per_extraction': total_input_tokens / total_extractions,
                'avg_extractions_per_submission': total_extractions / len(contents)
            },
            'submission_details': batch_estimates
        }
    
    def _get_input_token_cost(self) -> float:
        """Get input token cost per 1M tokens based on model."""
        # Adjust these based on current pricing
        model_costs = {
            "gpt-4.1": 2.00,
            "gpt-4.1-mini": 0.40, 
            "gpt-4.1-nano":  0.10, 
            "gpt-4o": 2.50,
            "gpt-4o-mini": 0.15,
        }
        
        # Check for exact match or partial match
        model_name = self.model_config.model_name
        for key, cost in model_costs.items():
            if key in model_name:
                return cost
        
        # Default cost if model not found
        return 1.00

    def _get_output_token_cost(self) -> float:
        """Get input token cost per 1M tokens based on model."""
        # Adjust these based on current pricing
        model_costs = {
            "gpt-4.1": 8.00,
            "gpt-4.1-mini": 1.60, 
            "gpt-4.1-nano":  0.40, 
            "gpt-4o": 10.00,
            "gpt-4o-mini": 0.6,
        }
        
        # Check for exact match or partial match
        model_name = self.model_config.model_name
        for key, cost in model_costs.items():
            if key in model_name:
                return cost
        
        # Default cost if model not found
        return 1.00

    
    def get_token_metrics(self) -> Dict[str, Any]:
        """Get token usage metrics."""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'avg_input_tokens_per_extraction': (
                self.total_input_tokens / self.extraction_count 
                if self.extraction_count > 0 else 0
            ),
            'avg_output_tokens_per_extraction': (
                self.total_output_tokens / self.extraction_count 
                if self.extraction_count > 0 else 0
            ),
            'estimated_total_cost': self._calculate_total_cost()
        }
    
    def _calculate_total_cost(self) -> str:
        """Calculate total cost based on token usage."""
        input_cost = (self.total_input_tokens / 1_000_000) * self._get_input_token_cost()
        output_cost = (self.total_output_tokens / 1_000_000) * self._get_output_token_cost()
        total_cost = input_cost + output_cost
        return f"${total_cost:.2f}"

class AsyncRateLimiter:
    """
    Async rate limiter to respect both API request and token limits.
    
    Tracks both requests per minute (RPM) and tokens per minute (TPM)
    to prevent hitting either limit.
    """
    
    def __init__(self, 
                 requests_per_minute: int, 
                 tokens_per_minute: int,
                 token_buffer_ratio: float = 0.9,
                 request_buffer_ratio: float = 0.9,
                 enabled: bool = True):
        """
        Initialize dual rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
            token_buffer_ratio: Use this fraction of token limit (safety buffer)
            request_buffer_ratio: Use this fraction of request limit (safety buffer)
            enabled: Whether rate limiting is enabled
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.enabled = enabled
        
        # Apply safety buffers
        self.effective_rpm = int(requests_per_minute * request_buffer_ratio)
        self.effective_tpm = int(tokens_per_minute * token_buffer_ratio)
        
        # Calculate minimum intervals
        self.min_request_interval = 60.0 / self.effective_rpm if self.effective_rpm > 0 else 0
        self.min_token_interval = 60.0 / self.effective_tpm if self.effective_tpm > 0 else 0
        
        # Tracking variables
        self.last_request_time = 0
        self.request_count = 0
        self.total_tokens_used = 0
        
        # Sliding window tracking (for more accurate rate limiting)
        self.request_timestamps = deque()  # Track request times in last minute
        self.token_usage = deque()  # Track (timestamp, tokens) in last minute
        
        self._lock = asyncio.Lock()
        
        logger.info(f"Rate limiter initialized:")
        logger.info(f"  Requests: {self.effective_rpm}/{requests_per_minute} RPM (buffer: {request_buffer_ratio:.1%})")
        logger.info(f"  Tokens: {self.effective_tpm:,}/{tokens_per_minute:,} TPM (buffer: {token_buffer_ratio:.1%})")
        logger.info(f"  Min intervals: {self.min_request_interval:.3f}s (req), {self.min_token_interval:.6f}s (token)")
    
    async def acquire(self, estimated_tokens: int = 1000) -> Dict[str, Any]:
        """
        Acquire permission to make a request with estimated token usage.
        
        Args:
            estimated_tokens: Estimated tokens this request will use
            
        Returns:
            Dictionary with timing information
        """
        if not self.enabled:
            return {'delay': 0, 'reason': 'disabled'}
        
        async with self._lock:
            current_time = time.time()
            
            # Clean old entries from sliding windows
            self._clean_old_entries(current_time)
            
            # Check both request and token limits
            request_delay = self._calculate_request_delay(current_time)
            token_delay = self._calculate_token_delay(current_time, estimated_tokens)
            
            # Use the longer delay
            total_delay = max(request_delay, token_delay)
            
            if total_delay > 0:
                delay_reason = "requests" if request_delay > token_delay else "tokens"
                logger.debug(f"Rate limiting: sleeping {total_delay:.3f}s due to {delay_reason}")
                await asyncio.sleep(total_delay)
                current_time = time.time()
            
            # Record this request
            self.last_request_time = current_time
            self.request_count += 1
            self.request_timestamps.append(current_time)
            self.token_usage.append((current_time, estimated_tokens))
            self.total_tokens_used += estimated_tokens
            
            return {
                'delay': total_delay,
                'reason': delay_reason if total_delay > 0 else 'no_delay',
                'current_rpm': len(self.request_timestamps),
                'current_tpm': sum(tokens for _, tokens in self.token_usage),
                'estimated_tokens': estimated_tokens
            }
    
    def _clean_old_entries(self, current_time: float):
        """Remove entries older than 1 minute from sliding windows"""
        cutoff_time = current_time - 60.0
        
        # Clean request timestamps
        while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
            self.request_timestamps.popleft()
        
        # Clean token usage
        while self.token_usage and self.token_usage[0][0] < cutoff_time:
            self.token_usage.popleft()
    
    def _calculate_request_delay(self, current_time: float) -> float:
        """Calculate delay needed to respect request rate limit"""
        # Simple interval-based check
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            return self.min_request_interval - time_since_last
        
        # Sliding window check
        if len(self.request_timestamps) >= self.effective_rpm:
            # If we're at the limit, wait until the oldest request is >1 minute old
            oldest_request = self.request_timestamps[0]
            time_until_oldest_expires = 60.0 - (current_time - oldest_request)
            if time_until_oldest_expires > 0:
                return time_until_oldest_expires
        
        return 0.0
    
    def _calculate_token_delay(self, current_time: float, estimated_tokens: int) -> float:
        """Calculate delay needed to respect token rate limit"""
        current_tokens_in_window = sum(tokens for _, tokens in self.token_usage)
        
        # Check if adding this request would exceed token limit
        if current_tokens_in_window + estimated_tokens > self.effective_tpm:
            # Find when enough tokens will "expire" from the window
            tokens_needed_to_free = (current_tokens_in_window + estimated_tokens) - self.effective_tpm
            tokens_freed = 0
            
            for timestamp, tokens in self.token_usage:
                tokens_freed += tokens
                if tokens_freed >= tokens_needed_to_free:
                    # Wait until this timestamp is >1 minute old
                    time_until_expires = 60.0 - (current_time - timestamp)
                    return max(0, time_until_expires)
            
            # If we can't find enough tokens to free, wait a full minute
            return 60.0
        
        return 0.0
    
    def update_actual_usage(self, actual_input_tokens: int, actual_output_tokens: int):
        """
        Update with actual token usage after request completion.
        
        This helps improve future estimates and tracking accuracy.
        
        Args:
            actual_input_tokens: Actual input tokens used
            actual_output_tokens: Actual output tokens used
        """
        if not self.enabled:
            return
        
        actual_total = actual_input_tokens + actual_output_tokens
        
        # Update the most recent token usage entry with actual values
        if self.token_usage:
            timestamp, estimated_tokens = self.token_usage[-1]
            # Replace estimated with actual
            self.token_usage[-1] = (timestamp, actual_total)
            
            # Update total tracking
            self.total_tokens_used = self.total_tokens_used - estimated_tokens + actual_total
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current usage within the sliding window"""
        current_time = time.time()
        self._clean_old_entries(current_time)
        
        current_requests = len(self.request_timestamps)
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        
        return {
            'current_rpm': current_requests,
            'current_tpm': current_tokens,
            'rpm_utilization': current_requests / self.effective_rpm if self.effective_rpm > 0 else 0,
            'tpm_utilization': current_tokens / self.effective_tpm if self.effective_tpm > 0 else 0,
            'rpm_remaining': max(0, self.effective_rpm - current_requests),
            'tpm_remaining': max(0, self.effective_tpm - current_tokens)
        }
    
    def can_make_request(self, estimated_tokens: int = 1000) -> Dict[str, Any]:
        """
        Check if a request can be made without waiting.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            Dictionary with availability information
        """
        current_time = time.time()
        self._clean_old_entries(current_time)
        
        request_delay = self._calculate_request_delay(current_time)
        token_delay = self._calculate_token_delay(current_time, estimated_tokens)
        total_delay = max(request_delay, token_delay)
        
        return {
            'can_proceed': total_delay == 0,
            'delay_needed': total_delay,
            'limited_by': 'requests' if request_delay > token_delay else 'tokens' if token_delay > 0 else None,
            'current_usage': self.get_current_usage()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive rate limiter statistics."""
        current_usage = self.get_current_usage()
        
        return {
            'total_requests': self.request_count,
            'total_tokens_used': self.total_tokens_used,
            'limits': {
                'requests_per_minute': self.requests_per_minute,
                'tokens_per_minute': self.tokens_per_minute,
                'effective_rpm': self.effective_rpm,
                'effective_tpm': self.effective_tpm
            },
            'current_usage': current_usage,
            'enabled': self.enabled,
            'intervals': {
                'min_request_interval': self.min_request_interval,
                'min_token_interval': self.min_token_interval
            }
        }


async def extract_sentiments_async(
    contents: List[RedditContent],
    model_config: Optional[ModelConfig] = None,
    async_config: Optional[AsyncExtractionConfig] = None,
    show_progress: bool = True
) -> Dict[str, List[SentimentResults]]:
    """
    Convenience function for async sentiment extraction.
    
    Args:
        contents: List of RedditContent objects
        model_config: Model configuration
        async_config: Async processing configuration
        show_progress: Whether to show progress
        
    Returns:
        Dictionary mapping submission_id to list of SentimentResults
    """
    extractor = AsyncSentimentExtractor(
        model_config=model_config,
        async_config=async_config
    )
    
    return await extractor.extract_batch_async(
        contents=contents,
        show_progress=show_progress
    )

