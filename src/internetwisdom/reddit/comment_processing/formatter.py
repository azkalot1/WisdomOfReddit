from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from .aggregator import CommentAggregator

logger = logging.getLogger(__name__)

class RedditCommentFormatter:
    """
    Formats Reddit submission and comments data into the specific JSON structure with metadata
    
    Single Responsibility: Data structure formatting and organization
    Uses CommentAggregator for comment threading logic
    """
    
    def __init__(self, comment_aggregator: CommentAggregator):
        """
        Initialize formatter with comment aggregator dependency
        
        Args:
            comment_aggregator: CommentAggregator instance for threading comments
        """
        self.comment_aggregator = comment_aggregator
    
    def format_submission_data(self, 
                            submission_data: Dict[str, Any], 
                            raw_comments: List[Dict[str, Any]],
                            include_metadata: bool = True) -> Dict[str, Any]:
        """
        Format submission and comments into the required JSON structure with metadata
        """
        # Format the date
        created_utc = submission_data.get('created_utc', 0)
        formatted_date = self._format_date(created_utc)
        date_for_path = self._format_date_for_path(created_utc)
        
        # Get submission body (handle both selftext and empty posts)
        submission_body = submission_data.get('selftext', '').strip()
        
        # Aggregate comments into threaded format
        comment_threads = self.comment_aggregator.aggregate_comments_to_threads(raw_comments)
        
        # Build the comments array starting with submission body
        comments_array = []
        comments_array.append(f"Submission body: {submission_body}")
        comments_array.extend(comment_threads)
        
        # Build the core formatted structure
        formatted_data = {
            "submission_id": submission_data.get('id', ''),
            "title": submission_data.get('title', ''),
            "date": formatted_date,
            "date_path": date_for_path,
            "subreddit": submission_data.get('subreddit', ''),
            "comments": comments_array
        }
        
        # Add metadata if requested
        if include_metadata:
            statistics = self._get_formatting_statistics(formatted_data, raw_comments)
            validation = self._validate_formatted_data(formatted_data)
            
            formatted_data.update({
                "metadata": {
                    "processing_timestamp": datetime.utcnow().isoformat(),
                    "statistics": statistics,
                    "validation": validation
                }
            })
        
        logger.info(f"Formatted submission {submission_data.get('id')} with {len(comment_threads)} comment threads")
        return formatted_data

    def _format_date_for_path(self, created_utc: float) -> str:
        """
        Format Unix timestamp to YYYYMMDD format for file paths
        
        Args:
            created_utc: Unix timestamp from Reddit API
            
        Returns:
            Date string in YYYYMMDD format: "20240405"
        """
        if not created_utc:
            return datetime.utcnow().strftime('%Y%m%d')  # Current date if no timestamp
        
        dt = datetime.utcfromtimestamp(created_utc)
        return dt.strftime('%Y%m%d')  # YYYYMMDD format
    
    def _format_date(self, created_utc: float) -> str:
        """
        Format Unix timestamp to ISO format
        
        Args:
            created_utc: Unix timestamp from Reddit API
            
        Returns:
            ISO formatted date string: "2024-04-05T09:01:14"
        """
        if not created_utc:
            return datetime.utcnow().isoformat()[:19]  # Current time if no timestamp
        
        dt = datetime.utcfromtimestamp(created_utc)
        return dt.isoformat()[:19]  # Remove microseconds, keep YYYY-MM-DDTHH:MM:SS
    
    def _get_formatting_statistics(self, formatted_data: Dict[str, Any], raw_comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the formatted data
        
        Args:
            formatted_data: Formatted submission data
            raw_comments: Original raw comments for additional stats
            
        Returns:
            Statistics about the formatted content
        """
        comments = formatted_data.get('comments', [])
        
        # Count different types of comments
        submission_body_count = 1 if len(comments) > 0 and comments[0].startswith("Submission body:") else 0
        comment_thread_count = len(comments) - submission_body_count
        
        # Get detailed comment statistics from aggregator
        comment_stats = self.comment_aggregator.get_comment_statistics(raw_comments)
        
        # Calculate score statistics
        comment_scores = [comment.get('score', 0) for comment in raw_comments]
        max_score = max(comment_scores) if comment_scores else 0
        min_score = min(comment_scores) if comment_scores else 0
        
        # Estimate replies by analyzing comment threads
        total_estimated_replies = 0
        for comment in comments[submission_body_count:]:  # Skip submission body
            if "; Replies: " in comment:
                replies_part = comment.split("; Replies: ")[1]
                # Count replies by counting score patterns {n}
                reply_count = replies_part.count(";") if replies_part.strip() else 0
                total_estimated_replies += reply_count
        
        # Text length analysis
        submission_body_text = comments[0][len("Submission body: "):] if submission_body_count > 0 else ""
        
        return {
            "raw_comment_count": len(raw_comments),
            "formatted_comment_entries": len(comments),
            "comment_threads": comment_thread_count,
            "submission_body_length": len(submission_body_text),
            "title_length": len(formatted_data.get('title', '')),
            "score_statistics": {
                "total_score": comment_stats.get('total_score', 0),
                "average_score": comment_stats.get('avg_score_per_comment', 0),
                "max_score": max_score,
                "min_score": min_score
            },
            "comment_breakdown": {
                "top_level_comments": comment_stats.get('top_level', 0),
                "total_replies": comment_stats.get('replies', 0),
                "empty_comments": comment_stats.get('empty', 0),
                "threads_with_replies": sum(1 for c in comments[1:] if "; Replies: " in c and c.split("; Replies: ")[1].strip())
            },
            "data_quality": {
                "comments_with_scores": len([c for c in raw_comments if 'score' in c]),
                "deleted_comments": len([c for c in raw_comments if c.get('body') in ['[deleted]', '[removed]', '']]),
                "processing_success_rate": round((len(raw_comments) - comment_stats.get('empty', 0)) / max(len(raw_comments), 1) * 100, 2)
            }
        }
    
    def _validate_formatted_data(self, formatted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the formatted data structure
        
        Args:
            formatted_data: Formatted submission data
            
        Returns:
            Validation results with any issues found
        """
        issues = []
        warnings = []
        
        # Check required fields
        required_fields = ['submission_id', 'title', 'date', 'subreddit', 'comments']
        for field in required_fields:
            if field not in formatted_data:
                issues.append(f"Missing required field: {field}")
            elif not formatted_data[field] and field not in ['title']:  # Title can be empty
                if field == 'comments':
                    issues.append(f"Empty required field: {field}")
                else:
                    warnings.append(f"Empty field: {field}")
        
        # Check comments structure
        comments = formatted_data.get('comments', [])
        if not isinstance(comments, list):
            issues.append("Comments field must be a list")
        elif len(comments) == 0:
            issues.append("Comments list is empty")
        else:
            # Check first entry is submission body
            if not comments[0].startswith("Submission body:"):
                issues.append("First comment entry should be submission body")
            
            # Check comment thread format
            malformed_threads = 0
            for i, comment in enumerate(comments[1:], 1):  # Skip submission body
                if not comment.startswith("Top level comment:"):
                    malformed_threads += 1
            
            if malformed_threads > 0:
                warnings.append(f"{malformed_threads} comment threads don't follow expected format")
        
        # Check date format
        date_str = formatted_data.get('date', '')
        try:
            datetime.fromisoformat(date_str)
        except ValueError:
            issues.append(f"Invalid date format: {date_str}")
        
        # Check submission ID format
        submission_id = formatted_data.get('submission_id', '')
        if submission_id and (len(submission_id) < 5 or len(submission_id) > 10):
            warnings.append(f"Unusual submission ID length: {submission_id}")
        
        return {
            "is_valid": len(issues) == 0,
            "is_clean": len(issues) == 0 and len(warnings) == 0,
            "issues": issues,
            "warnings": warnings,
            "validation_timestamp": datetime.utcnow().isoformat()[:19]
        }
    
    def format_multiple_submissions(self, 
                                  submissions_data: List[Dict[str, Any]], 
                                  comments_data: Dict[str, List[Dict[str, Any]]],
                                  include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Format multiple submissions at once
        
        Args:
            submissions_data: List of submission data dictionaries
            comments_data: Dictionary mapping submission_id -> comments list
            include_metadata: Whether to include metadata in each submission
            
        Returns:
            List of formatted submission dictionaries with metadata
        """
        formatted_submissions = []
        
        for submission in submissions_data:
            submission_id = submission.get('id', '')
            submission_comments = comments_data.get(submission_id, [])
            
            formatted_submission = self.format_submission_data(
                submission, 
                submission_comments, 
                include_metadata=include_metadata
            )
            formatted_submissions.append(formatted_submission)
        
        logger.info(f"Formatted {len(formatted_submissions)} submissions")
        return formatted_submissions