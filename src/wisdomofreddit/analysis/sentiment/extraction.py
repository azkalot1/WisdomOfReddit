from .state import ModelConfig, SentimentResults, RedditContent, ExtractionError
from .prompts import extractor_system_prompt, extractor_content
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel
import tiktoken
import logging
from tenacity import retry, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)

class SentimentExtractor:
    """
    Extracts sentiment from Reddit submissions and comments using LLM.
    
    Processes each comment individually with submission context to avoid
    overcounting submission sentiment.
    """
    
    def __init__(
        self, 
        model_config: Optional[ModelConfig] = None,
        system_prompt: Optional[str] = None,
        content_template: Optional[str] = None,
        output_model: Optional[BaseModel] = None
    ):
        """
        Initialize the sentiment extractor.
        
        Args:
            model_config: Configuration for the LLM model
            system_prompt: System prompt for extraction (uses default if None)
            content_template: Template for formatting content (uses default if None)
        """
        self.model_config = model_config or ModelConfig()
        self.system_prompt = system_prompt or extractor_system_prompt
        self.content_template = content_template or extractor_content
        self.output_model = output_model or SentimentResults
        # Initialize model with structured output
        self.model = self._initialize_model()
        
        # Initialize tokenizer for accurate token counting
        self.encoding = self._get_encoding()
        
        # Track metrics
        self.extraction_count = 0
        self.error_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        logger.info(f"Initialized SentimentExtractor with model: {self.model_config.model_name}")
    
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
    
    def extract_all(self, content: RedditContent) -> List[SentimentResults]:
        """
        Extract sentiment from submission and each comment individually.
        
        This method processes:
        1. Submission alone (if no comments)
        2. Submission + each comment individually
        
        Args:
            content: RedditContent object containing submission and comments
            
        Returns:
            List of SentimentResults, one for each extraction
        """
        results = []
        
        # If no comments, process submission alone
        if not content.comments:
            logger.debug(f"Processing submission alone (no comments) for {content.submission_id}")
            result = self._extract_single(
                title=content.title,
                submission_body=content.submission_body,
                comment_text="",
                extraction_context="submission_only"
            )
            results.append(result)
        else:
            # Process submission + each comment
            for i, comment in enumerate(content.comments):
                logger.debug(f"Processing submission + comment {i+1}/{len(content.comments)}")
                result = self._extract_single(
                    title=content.title,
                    submission_body=content.submission_body,
                    comment_text=comment,
                    extraction_context=f"comment_{i+1}"
                )
                results.append(result)
        
        logger.info(
            f"Completed extraction for {content.submission_id}: "
            f"{len(results)} extractions performed"
        )
        
        return results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _extract_single(
        self, 
        title: str, 
        submission_body: str, 
        comment_text: str,
        extraction_context: str = ""
    ) -> SentimentResults:
        """
        Extract sentiment from a single submission + comment pair.
        
        Args:
            title: Submission title
            submission_body: Submission body text
            comment_text: Single comment text (empty string for submission only)
            extraction_context: Context identifier for logging
            
        Returns:
            SentimentResults object with extracted sentiments
        """
        # Format content
        formatted_content = self._format_single_extraction(
            title, 
            submission_body, 
            comment_text
        )
        
        try:
            # Create messages for LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=formatted_content)
            ]
            
            # Extract sentiment
            start_time = datetime.now()
            response = self.model.invoke(messages)
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            # Validate response
            if not isinstance(response, SentimentResults):
                raise ExtractionError(f"Invalid response type: {type(response)}")
            
            # Post-process results
            response = self._post_process_results(response)
            
            # Add extraction context metadata
            if hasattr(response, 'extraction_context'):
                response.extraction_context = extraction_context
            
            # Update metrics
            self.extraction_count += 1
            logger.debug(
                f"Extraction {extraction_context} completed in {extraction_time:.2f}s. "
                f"Found {len(response.sentiments)} sentiments."
            )
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Extraction failed for {extraction_context}: {e}")
            raise ExtractionError(f"Extraction failed: {e}") from e
    
    def _format_single_extraction(
        self, 
        title: str, 
        submission_body: str, 
        comment_text: str
    ) -> str:
        """
        Format content for a single extraction.
        
        Args:
            title: Submission title
            submission_body: Submission body
            comment_text: Single comment (or empty string)
            
        Returns:
            Formatted string for LLM
        """
        # Handle case where there's no comment
        if not comment_text:
            comments_section = "No comments"
        else:
            comments_section = comment_text
        
        # Format using template
        formatted = self.content_template.format(
            title_text=title,
            submission_text=submission_body or "No submission body",
            comments_text=comments_section
        )

        return formatted
    
    def extract_batch(
        self, 
        contents: List[RedditContent], 
        continue_on_error: bool = True
    ) -> Dict[str, List[SentimentResults]]:
        """
        Extract sentiment from multiple Reddit contents.
        
        Args:
            contents: List of RedditContent objects
            continue_on_error: Whether to continue processing on errors
            
        Returns:
            Dictionary mapping submission_id to list of SentimentResults
        """
        all_results = {}
        
        for content in contents:
            submission_id = content.submission_id or f"submission_{id(content)}"
            
            try:
                results = self.extract_all(content)
                all_results[submission_id] = results
            except Exception as e:
                logger.error(f"Failed to process submission {submission_id}: {e}")
                if continue_on_error:
                    all_results[submission_id] = []
                else:
                    raise
        
        # Log summary
        total_submissions = len(contents)
        successful_submissions = sum(1 for results in all_results.values() if results)
        total_extractions = sum(len(results) for results in all_results.values())
        
        logger.info(
            f"Batch extraction complete: {successful_submissions}/{total_submissions} "
            f"submissions processed, {total_extractions} total extractions"
        )
        
        return all_results
    
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get extraction metrics."""
        return {
            'total_extractions': self.extraction_count,
            'total_errors': self.error_count,
            'success_rate': (
                self.extraction_count / (self.extraction_count + self.error_count)
                if (self.extraction_count + self.error_count) > 0 
                else 0
            ),
            'model': self.model_config.model_name
        }

    def _get_encoding(self):
        """Get the appropriate encoding for the model."""
        model_name = self.model_config.model_name
        
        # Map model names to encodings
        if "gpt-4" in model_name:
            return tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model_name:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            # Default to cl100k_base encoding (used by most recent models)
            return tiktoken.get_encoding("cl100k_base")
    
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
