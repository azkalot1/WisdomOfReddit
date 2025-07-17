import praw
import functools
import time
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Exceptions
class RedditAPIError(Exception):
    """Base exception for Reddit API errors"""
    pass

class RateLimitError(RedditAPIError):
    """Raised when rate limit is exceeded"""
    pass

class AuthenticationError(RedditAPIError):
    """Raised when authentication fails"""
    pass

class SubmissionNotFoundError(RedditAPIError):
    """Raised when submission is not found"""
    pass

# Configuration
@dataclass
class RedditConfig:
    client_id: str
    client_secret: str
    user_agent: str = "reddit_api_client_v1.0"
    rate_limit_seconds: int = 300
    max_retries: int = 5
    retry_delay: int = 15
    exponential_backoff: bool = True
    timeout: int = 30

# Enhanced Retry Decorator
def robust_retry(max_attempts: int = 5, base_delay: float = 15, exponential_backoff: bool = True):
    """
    Robust retry decorator with exponential backoff and specific exception handling
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (AuthenticationError, SubmissionNotFoundError) as e:
                    # Don't retry on these specific errors
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
                except Exception as e:
                    last_exception = e
                    
                    # Calculate delay
                    if exponential_backoff:
                        delay = min(base_delay * (2 ** attempt), 300)  # Cap at 5 minutes
                    else:
                        delay = base_delay
                    
                    # Log the error
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
                    )
                    
                    # Don't sleep on the last attempt
                    if attempt < max_attempts - 1:
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            # If we get here, all attempts failed
            raise RedditAPIError(f"Failed after {max_attempts} attempts: {last_exception}")
        return wrapper
    return decorator

# Abstract Interface
class RedditAPIClient(ABC):
    """Abstract interface for Reddit API clients"""
    
    @abstractmethod
    def get_submission(self, submission_id: str) -> Dict[str, Any]:
        """Fetch single submission data"""
        pass
    
    @abstractmethod
    def get_submissions_by_timeframe(self, subreddit: str, hours: int, limit: Optional[int] = None, **kwargs) -> List[str]:
        """Fetch submission IDs within timeframe"""
        pass
    
    @abstractmethod  
    def get_comments(self, submission_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch raw comments for submission"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test API connectivity"""
        pass

# Concrete Implementation
class PrawRedditClient(RedditAPIClient):
    """
    Pure Reddit API client using PRAW - handles only API interactions
    Follows Single Responsibility Principle: only API calls, no processing
    """
    
    def __init__(self, config: RedditConfig):
        """
        Initialize Reddit API client
        
        Args:
            config: Reddit configuration object
        """
        self._config = config
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the PRAW Reddit client"""
        try:
            self._client = praw.Reddit(
                client_id=self._config.client_id,
                client_secret=self._config.client_secret,
                user_agent=self._config.user_agent,
                ratelimit_seconds=self._config.rate_limit_seconds,
                timeout=self._config.timeout
            )
            self._client.read_only = True
            
            logger.info("PRAW Reddit client initialized successfully")
            
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize Reddit client: {e}")
    
    @property
    def client(self):
        """Get the PRAW client instance"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    @robust_retry(max_attempts=3, base_delay=5, exponential_backoff=False)
    def test_connection(self) -> bool:
        """
        Test Reddit API connectivity
        
        Returns:
            True if connection is successful
            
        Raises:
            RedditAPIError: If connection test fails
        """
        try:
            # Simple API call to test connectivity
            test_subreddit = self.client.subreddit('test')
            list(test_subreddit.hot(limit=1))
            logger.info("Reddit API connection test successful")
            return True
            
        except Exception as e:
            raise RedditAPIError(f"Reddit API connection test failed: {e}")
    
    @robust_retry(max_attempts=5, base_delay=15)
    def get_submission(self, submission_id: str) -> Dict[str, Any]:
        """
        Fetch single submission data
        
        Args:
            submission_id: Reddit submission ID
            
        Returns:
            Dictionary containing submission data
            
        Raises:
            SubmissionNotFoundError: If submission doesn't exist
            RedditAPIError: For other API errors
        """
        if not submission_id or not isinstance(submission_id, str):
            raise ValueError("Submission ID must be a non-empty string")
        
        try:
            submission = self.client.submission(submission_id)
            
            # Access title to trigger API call and check if submission exists
            _ = submission.title
            
            return {
                'id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext,
                'created_utc': submission.created_utc,
                'score': submission.score,
                'upvote_ratio': getattr(submission, 'upvote_ratio', None),
                'num_comments': submission.num_comments,
                'subreddit': str(submission.subreddit),
                'author': str(submission.author) if submission.author else '[deleted]',
                'url': submission.url,
                'permalink': submission.permalink,
                'is_self': submission.is_self,
                'locked': submission.locked,
                'stickied': submission.stickied,
                'over_18': submission.over_18,
                'spoiler': submission.spoiler,
                'archived': submission.archived
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'not found' in error_msg or '404' in error_msg:
                raise SubmissionNotFoundError(f"Submission {submission_id} not found")
            else:
                raise RedditAPIError(f"Failed to fetch submission {submission_id}: {e}")
    
    @robust_retry(max_attempts=5, base_delay=15)
    def get_submissions_by_timeframe(self, subreddit: str, hours: int, limit: Optional[int] = None, inverse_order: bool = False) -> List[str]:
        """
        Fetch submission IDs within specified timeframe
        
        Args:
            subreddit: Subreddit name (without r/ prefix)
            hours: Number of hours back to search
            limit: Maximum number of submissions to check (None for no limit)
            inverse_order: If True, fetch submissions in reverse order (oldest first)
            
        Returns:
            List of submission IDs within the timeframe
            
        Raises:
            RedditAPIError: For API errors
        """
        if not subreddit or not isinstance(subreddit, str):
            raise ValueError("Subreddit name must be a non-empty string")
        
        if hours <= 0:
            raise ValueError("Hours must be a positive number")
        
        try:
            subreddit_obj = self.client.subreddit(subreddit)
            current_time = datetime.utcnow()
            time_threshold = current_time - timedelta(hours=hours)
            
            submission_ids = []
            processed_count = 0
            
            logger.info(f"Fetching submissions from r/{subreddit} from last {hours} hours")
            
            for submission in subreddit_obj.new(limit=limit):
                processed_count += 1
                submission_time = datetime.utcfromtimestamp(submission.created_utc)
                
                if submission_time >= time_threshold:
                    submission_ids.append(submission.id)
                    logger.debug(f"Added submission {submission.id} (created: {submission_time})")
                else:
                    # Break early if we've gone past our time threshold
                    logger.debug(f"Reached time threshold at submission {submission.id} (created: {submission_time})")
                    break
            
            logger.info(f"Found {len(submission_ids)} submissions in r/{subreddit} from last {hours} hours (processed {processed_count} total)")
            if inverse_order:
                submission_ids = submission_ids[::-1]  # Reverse the order
            return submission_ids
            
        except Exception as e:
            if 'subreddit not found' in str(e).lower():
                raise RedditAPIError(f"Subreddit r/{subreddit} not found or is private")
            else:
                raise RedditAPIError(f"Failed to fetch submissions from r/{subreddit}: {e}")
    
    @robust_retry(max_attempts=5, base_delay=15)  
    def get_comments(self, submission_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch raw comments for a submission
        
        Args:
            submission_id: Reddit submission ID
            limit: Maximum number of MoreComments objects to expand (None for no limit)
            
        Returns:
            List of raw comment dictionaries
            
        Raises:
            SubmissionNotFoundError: If submission doesn't exist
            RedditAPIError: For other API errors
        """
        if not submission_id or not isinstance(submission_id, str):
            raise ValueError("Submission ID must be a non-empty string")
        
        try:
            submission = self.client.submission(submission_id)
            
            # Test if submission exists by accessing a property
            _ = submission.title
            
            # Replace "MoreComments" objects with actual comments
            # This is the expensive operation that loads all comments
            if limit is not None:
                submission.comments.replace_more(limit=limit)
            else:
                submission.comments.replace_more(limit=0)  # Load all
            
            comments = []
            
            # Extract all comments (including nested replies) into a flat list
            for comment in submission.comments.list():
                if hasattr(comment, 'body'):  # Ensure it's a Comment, not MoreComments
                    comment_data = {
                        'id': comment.id,
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'is_submitter': comment.is_submitter,
                        'parent_id': comment.parent_id,
                        'permalink': comment.permalink,
                        'edited': comment.edited,
                        'gilded': getattr(comment, 'gilded', 0),
                        'stickied': comment.stickied,
                        'archived': comment.archived,
                        'locked': getattr(comment, 'locked', False),
                        'controversiality': getattr(comment, 'controversiality', 0)
                    }
                    comments.append(comment_data)
            
            logger.info(f"Fetched {len(comments)} comments for submission {submission_id}")
            return comments
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'not found' in error_msg or '404' in error_msg:
                raise SubmissionNotFoundError(f"Submission {submission_id} not found")
            else:
                raise RedditAPIError(f"Failed to fetch comments for submission {submission_id}: {e}")
    
    def get_subreddit_info(self, subreddit: str) -> Dict[str, Any]:
        """
        Get basic subreddit information
        
        Args:
            subreddit: Subreddit name (without r/ prefix)
            
        Returns:
            Dictionary containing subreddit information
            
        Raises:
            RedditAPIError: For API errors
        """
        if not subreddit or not isinstance(subreddit, str):
            raise ValueError("Subreddit name must be a non-empty string")
        
        try:
            subreddit_obj = self.client.subreddit(subreddit)
            
            return {
                'name': subreddit_obj.display_name,
                'title': subreddit_obj.title,
                'subscribers': subreddit_obj.subscribers,
                'active_user_count': getattr(subreddit_obj, 'active_user_count', None),
                'public_description': subreddit_obj.public_description,
                'description': subreddit_obj.description,
                'created_utc': subreddit_obj.created_utc,
                'over18': subreddit_obj.over18,
                'lang': getattr(subreddit_obj, 'lang', None),
                'quarantine': getattr(subreddit_obj, 'quarantine', False),
                'user_is_banned': getattr(subreddit_obj, 'user_is_banned', None),
                'user_is_subscriber': getattr(subreddit_obj, 'user_is_subscriber', None)
            }
            
        except Exception as e:
            if 'subreddit not found' in str(e).lower():
                raise RedditAPIError(f"Subreddit r/{subreddit} not found or is private")
            else:
                raise RedditAPIError(f"Failed to get subreddit info for r/{subreddit}: {e}")

# Configuration Factory
class RedditConfigFactory:
    """Factory for creating Reddit configuration objects"""
    
    @staticmethod
    def from_environment(env_path: Optional[Path] = None) -> RedditConfig:
        """
        Load Reddit configuration from environment variables
        
        Args:
            env_path: Path to .env file (optional)
            
        Returns:
            RedditConfig object
            
        Raises:
            AuthenticationError: If required credentials are missing
        """
        if env_path:
            load_dotenv(dotenv_path=env_path)
        else:
            # Default path - 2 levels up from current file
            env_path = Path(__file__).resolve().parents[2] / '.env'
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
        
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            raise AuthenticationError(
                "Reddit API credentials not found in environment variables. "
                "Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET"
            )
        
        return RedditConfig(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=os.getenv("REDDIT_USER_AGENT", "reddit_api_client_v1.0"),
            rate_limit_seconds=int(os.getenv("REDDIT_RATE_LIMIT_SECONDS", "300")),
            max_retries=int(os.getenv("REDDIT_MAX_RETRIES", "5")),
            retry_delay=int(os.getenv("REDDIT_RETRY_DELAY", "15")),
            exponential_backoff=os.getenv("REDDIT_EXPONENTIAL_BACKOFF", "true").lower() == "true",
            timeout=int(os.getenv("REDDIT_TIMEOUT", "30"))
        )