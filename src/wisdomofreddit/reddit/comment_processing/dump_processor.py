import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import logging
from urllib.parse import urlparse
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CSVConverterConfig:
    """Configuration for CSV to JSON conversion"""
    input_directory: Path
    output_directory: Path
    subreddits: List[str]  # List of subreddit names to process
    date_format: str = '%Y-%m-%d %H:%M'  # Format in CSV
    output_date_format: str = '%Y-%m-%dT%H:%M:%S'  # ISO format for output
    folder_date_format: str = '%Y%m%d'  # Format for folder names (YYYYMMDD)
    max_comments_per_submission: int = 1000
    include_submission_body_in_comments: bool = True
    remove_orphan_comments: bool = True  # Remove comments without corresponding submissions
    organize_by_subreddit: bool = True  # Create subreddit subfolders
    include_scores_in_text: bool = True  # Include scores in comment text
    min_year: Optional[int] = 2020  # Minimum year to include (None for no filter)
    max_year: Optional[int] = None  # Maximum year to include (None for no filter)

class RedditCSVConverter:
    """
    Converts Reddit CSV dumps back to JSON format for sentiment analysis pipeline
    
    Processes subreddit_submissions.csv and subreddit_comments.csv files
    and creates JSON files organized by date: YYYYMMDD/submission_name.json
    """
    
    def __init__(self, config: CSVConverterConfig):
        self.config = config
        self.stats = {
            'subreddits_processed': 0,
            'submissions_processed': 0,
            'comments_processed': 0,
            'files_created': 0,
            'errors': 0,
            'submissions_removed': 0,
            'comments_removed': 0,
            'orphan_comments_removed': 0,
            'date_folders_created': 0,
            'date_parse_errors': 0,
            'submissions_filtered_by_year': 0,
            'comments_filtered_by_year': 0
        }
        self.created_folders = set()  # Track created folders to avoid duplicate logging
    
    def convert_all_subreddits(self) -> Dict[str, Any]:
        """
        Convert all configured subreddits from CSV to JSON
        
        Returns:
            Dictionary with conversion statistics
        """
        logger.info(f"Starting conversion of {len(self.config.subreddits)} subreddits")
        
        # Log year filtering configuration
        if self.config.min_year or self.config.max_year:
            year_filter_msg = f"Year filtering: "
            if self.config.min_year:
                year_filter_msg += f"from {self.config.min_year}"
            if self.config.max_year:
                if self.config.min_year:
                    year_filter_msg += f" to {self.config.max_year}"
                else:
                    year_filter_msg += f"up to {self.config.max_year}"
            logger.info(year_filter_msg)
        else:
            logger.info("No year filtering applied")
        
        # Ensure output directory exists
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        
        for subreddit in self.config.subreddits:
            try:
                logger.info(f"Processing subreddit: {subreddit}")
                self._convert_subreddit(subreddit)
                self.stats['subreddits_processed'] += 1
            except Exception as e:
                logger.error(f"Failed to process subreddit {subreddit}: {e}")
                self.stats['errors'] += 1
        
        logger.info(f"Conversion complete: {self.stats}")
        return self.stats
    
    def _convert_subreddit(self, subreddit: str):
        """Convert a single subreddit's CSV files to JSON"""
        
        # File paths
        submissions_file = self.config.input_directory / f"{subreddit}_submissions.csv"
        comments_file = self.config.input_directory / f"{subreddit}_comments.csv"
        
        # Verify files exist
        if not submissions_file.exists():
            logger.error(f"Submissions file not found: {submissions_file}")
            return
        
        if not comments_file.exists():
            logger.error(f"Comments file not found: {comments_file}")
            return
        
        # Load CSV files
        logger.debug(f"Loading CSV files for {subreddit}")
        submissions_df = pd.read_csv(submissions_file)
        comments_df = pd.read_csv(comments_file)
        
        logger.info(f"Loaded {len(submissions_df)} submissions and {len(comments_df)} comments for {subreddit}")
        
        # Apply year filtering first
        submissions_df, comments_df = self._filter_by_year(submissions_df, comments_df, subreddit)
        
        logger.info(f"After year filtering: {len(submissions_df)} submissions and {len(comments_df)} comments for {subreddit}")
        
        # Clean the data
        submissions_df, comments_df = self._clean_data(submissions_df, comments_df, subreddit)
        
        logger.info(f"After cleaning: {len(submissions_df)} submissions and {len(comments_df)} comments for {subreddit}")
        
        # Group comments by submission
        comments_by_submission = self._group_comments_by_submission(comments_df)
        
        # Process each submission
        for _, submission_row in submissions_df.iterrows():
            try:
                json_data = self._convert_submission_to_json(
                    submission_row, 
                    comments_by_submission, 
                    subreddit
                )
                
                if json_data:
                    self._save_json_file_organized(json_data, subreddit, submission_row['created'])
                    self.stats['submissions_processed'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to convert submission {submission_row.get('link', 'unknown')}: {e}")
                self.stats['errors'] += 1
    
    def _filter_by_year(self, submissions_df: pd.DataFrame, comments_df: pd.DataFrame, subreddit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter submissions and comments by year range
        
        Args:
            submissions_df: DataFrame of submissions
            comments_df: DataFrame of comments
            subreddit: Subreddit name for logging
            
        Returns:
            Tuple of filtered DataFrames
        """
        if not self.config.min_year and not self.config.max_year:
            # No filtering needed
            return submissions_df, comments_df
        
        logger.debug(f"Applying year filtering for {subreddit}")
        
        original_submissions_count = len(submissions_df)
        original_comments_count = len(comments_df)
        
        # Filter submissions by year
        submissions_filtered = self._filter_dataframe_by_year(submissions_df, 'created')
        submissions_filtered_count = original_submissions_count - len(submissions_filtered)
        self.stats['submissions_filtered_by_year'] += submissions_filtered_count
        
        # Filter comments by year
        comments_filtered = self._filter_dataframe_by_year(comments_df, 'created')
        comments_filtered_count = original_comments_count - len(comments_filtered)
        self.stats['comments_filtered_by_year'] += comments_filtered_count
        
        logger.info(f"Year filtering for {subreddit}: removed {submissions_filtered_count} submissions and {comments_filtered_count} comments")
        
        return submissions_filtered, comments_filtered
    
    def _filter_dataframe_by_year(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Filter a DataFrame by year range based on a date column
        
        Args:
            df: DataFrame to filter
            date_column: Name of the date column
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying original
        df_filtered = df.copy()
        
        # Parse dates and extract years
        valid_rows = []
        
        for idx, row in df_filtered.iterrows():
            try:
                date_str = row[date_column]
                if pd.isna(date_str):
                    continue
                
                # Parse the date
                parsed_date = datetime.strptime(str(date_str), self.config.date_format)
                year = parsed_date.year
                
                # Check year constraints
                year_valid = True
                
                if self.config.min_year and year < self.config.min_year:
                    year_valid = False
                
                if self.config.max_year and year > self.config.max_year:
                    year_valid = False
                
                if year_valid:
                    valid_rows.append(idx)
                    
            except Exception as e:
                logger.warning(f"Failed to parse date {row[date_column]} for year filtering: {e}")
                # Include rows with unparseable dates (they'll be handled later)
                valid_rows.append(idx)
        
        # Return filtered DataFrame
        return df_filtered.loc[valid_rows] if valid_rows else pd.DataFrame(columns=df_filtered.columns)
    
    def _clean_data(self, submissions_df: pd.DataFrame, comments_df: pd.DataFrame, subreddit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean submissions and comments data by removing [removed] and [deleted] content
        and optionally removing orphan comments
        """
        original_submissions_count = len(submissions_df)
        original_comments_count = len(comments_df)
        
        # Clean submissions
        logger.debug(f"Cleaning submissions for {subreddit}")
        
        # Handle NaN values by filling with empty string for the filtering
        submissions_df['title'] = submissions_df['title'].fillna('')
        submissions_df['text'] = submissions_df['text'].fillna('')
        
        # Remove submissions with [removed] or [deleted] in title or text
        submissions_cleaned = submissions_df.loc[
            (~submissions_df.title.isin(['[removed]', '[deleted]'])) & 
            (~submissions_df.text.isin(['[removed]', '[deleted]'])), :
        ].copy()
        
        submissions_removed = original_submissions_count - len(submissions_cleaned)
        self.stats['submissions_removed'] += submissions_removed
        
        logger.info(f"Removed {submissions_removed} submissions with [removed]/[deleted] content from {subreddit}")
        
        # Clean comments
        logger.debug(f"Cleaning comments for {subreddit}")
        
        # Handle NaN values
        comments_df['body'] = comments_df['body'].fillna('')
        
        # Remove comments with [removed] or [deleted] body
        comments_cleaned = comments_df.loc[
            ~comments_df.body.isin(['[removed]', '[deleted]']), :
        ].copy()
        
        comments_removed = original_comments_count - len(comments_cleaned)
        self.stats['comments_removed'] += comments_removed
        
        logger.info(f"Removed {comments_removed} comments with [removed]/[deleted] content from {subreddit}")
        
        # Remove orphan comments if configured
        if self.config.remove_orphan_comments:
            comments_cleaned = self._remove_orphan_comments(submissions_cleaned, comments_cleaned, subreddit)
        
        return submissions_cleaned, comments_cleaned
    
    def _remove_orphan_comments(self, submissions_df: pd.DataFrame, comments_df: pd.DataFrame, subreddit: str) -> pd.DataFrame:
        """
        Remove comments that don't have corresponding submissions
        """
        logger.debug(f"Removing orphan comments for {subreddit}")
        
        # Extract submission IDs from submissions
        valid_submission_ids = set()
        for _, submission_row in submissions_df.iterrows():
            submission_id = self._extract_submission_id_from_submission_link(submission_row['link'])
            if submission_id:
                valid_submission_ids.add(submission_id)
        
        logger.debug(f"Found {len(valid_submission_ids)} valid submission IDs for {subreddit}")
        
        # Filter comments to only include those with valid submission IDs
        original_comment_count = len(comments_df)
        valid_comments = []
        
        for _, comment_row in comments_df.iterrows():
            comment_submission_id = self._extract_submission_id_from_comment_link(comment_row['link'])
            if comment_submission_id and comment_submission_id in valid_submission_ids:
                valid_comments.append(comment_row)
        
        if valid_comments:
            comments_cleaned = pd.DataFrame(valid_comments)
        else:
            # Create empty DataFrame with same columns
            comments_cleaned = pd.DataFrame(columns=comments_df.columns)
        
        orphans_removed = original_comment_count - len(comments_cleaned)
        self.stats['orphan_comments_removed'] += orphans_removed
        
        logger.info(f"Removed {orphans_removed} orphan comments from {subreddit}")
        
        return comments_cleaned
    
    def _group_comments_by_submission(self, comments_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Group comments by their parent submission"""
        
        comments_by_submission = {}
        
        for _, comment_row in comments_df.iterrows():
            # Extract submission ID from comment link
            submission_id = self._extract_submission_id_from_comment_link(comment_row['link'])
            
            if submission_id:
                if submission_id not in comments_by_submission:
                    comments_by_submission[submission_id] = []
                
                comment_data = {
                    'author': comment_row['author'],
                    'score': comment_row['score'],
                    'created': comment_row['created'],
                    'body': comment_row['body'],
                    'link': comment_row['link']
                }
                
                comments_by_submission[submission_id].append(comment_data)
                self.stats['comments_processed'] += 1
        
        logger.debug(f"Grouped comments into {len(comments_by_submission)} submissions")
        return comments_by_submission
    
    def _extract_submission_id_from_comment_link(self, comment_link: str) -> Optional[str]:
        """
        Extract submission ID from comment link
        
        Example: https://www.reddit.com/r/ai_trading/comments/iqn31w/title/g4v5dr8/
        Returns: iqn31w
        """
        try:
            if pd.isna(comment_link):
                return None
                
            # Parse URL path
            path = urlparse(str(comment_link)).path
            # Split path and find submission ID (after /comments/)
            path_parts = path.split('/')
            
            if 'comments' in path_parts:
                comments_index = path_parts.index('comments')
                if comments_index + 1 < len(path_parts):
                    return path_parts[comments_index + 1]
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract submission ID from {comment_link}: {e}")
            return None
    
    def _extract_submission_id_from_submission_link(self, submission_link: str) -> Optional[str]:
        """
        Extract submission ID from submission link
        
        Example: https://www.reddit.com/r/ai_trading/comments/ikibe8/title/
        Returns: ikibe8
        """
        try:
            if pd.isna(submission_link):
                return None
                
            path = urlparse(str(submission_link)).path
            path_parts = path.split('/')
            
            if 'comments' in path_parts:
                comments_index = path_parts.index('comments')
                if comments_index + 1 < len(path_parts):
                    return path_parts[comments_index + 1]
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract submission ID from {submission_link}: {e}")
            return None
    
    def _calculate_comment_stats(self, raw_comments: List[Dict], processed_comments: List[str]) -> Dict[str, Any]:
        """Calculate detailed comment statistics"""
        
        stats = {
            'top_level': 0,
            'replies': 0,
            'empty': 0,
            'with_scores': 0,
            'deleted_removed': 0,
            'valid_processed': len(processed_comments)
        }
        
        for comment in raw_comments:
            # Count comments with scores
            if pd.notna(comment.get('score')):
                stats['with_scores'] += 1
            
            # Count deleted/removed comments
            body = str(comment.get('body', '')).strip()
            if body in ['[deleted]', '[removed]', '', 'nan']:
                stats['deleted_removed'] += 1
                stats['empty'] += 1
            else:
                stats['top_level'] += 1
        
        # For now, we're treating all comments as top-level since CSV doesn't have reply structure
        # In a full Reddit API dump, you'd have parent_id to determine reply structure
        
        return stats
    
    def _format_comment_with_score(self, comment_text: str, score: Any, comment_type: str = "Top level comment") -> str:
        """Format comment text with score"""
        
        if not self.config.include_scores_in_text:
            return f"{comment_type}: {comment_text}"
        
        # Handle score formatting
        try:
            score_int = int(score) if pd.notna(score) else 0
            score_str = f"(score {score_int})"
        except (ValueError, TypeError):
            score_str = "(score unknown)"
        
        return f"{comment_type}: {comment_text} {score_str}"
    
    def _convert_submission_to_json(self, submission_row: pd.Series, 
                                   comments_by_submission: Dict[str, List[Dict]], 
                                   subreddit: str) -> Optional[Dict[str, Any]]:
        """Convert a single submission row to JSON format"""
        
        # Extract submission ID
        submission_id = self._extract_submission_id_from_submission_link(submission_row['link'])
        
        if not submission_id:
            logger.warning(f"Could not extract submission ID from {submission_row['link']}")
            return None
        
        # Parse date
        try:
            created_date = datetime.strptime(submission_row['created'], self.config.date_format)
            formatted_date = created_date.strftime(self.config.output_date_format)
        except Exception as e:
            logger.warning(f"Failed to parse date {submission_row['created']}: {e}")
            self.stats['date_parse_errors'] += 1
            formatted_date = submission_row['created']
        
        # Get comments for this submission
        raw_submission_comments = comments_by_submission.get(submission_id, [])
        
        # Limit comments if configured
        if len(raw_submission_comments) > self.config.max_comments_per_submission:
            # Sort by score (descending) and take top comments
            raw_submission_comments = sorted(raw_submission_comments, key=lambda x: x.get('score', 0), reverse=True)
            raw_submission_comments = raw_submission_comments[:self.config.max_comments_per_submission]
            logger.debug(f"Limited comments to {self.config.max_comments_per_submission} for submission {submission_id}")
        
        # Build comments list in expected format
        comments_list = []
        
        # Add submission body as first "comment" if configured and exists
        if self.config.include_submission_body_in_comments and pd.notna(submission_row.get('text', '')):
            submission_text = str(submission_row['text']).strip()
            if submission_text and submission_text != 'nan' and submission_text not in ['[removed]', '[deleted]']:
                cleaned_text = self._clean_text(submission_text)
                if cleaned_text:
                    # Format submission body with score
                    submission_score = submission_row.get('score', 0)
                    formatted_submission = self._format_comment_with_score(
                        cleaned_text, 
                        submission_score, 
                        "Submission body"
                    )
                    comments_list.append(formatted_submission)
        
        # Add actual comments
        for comment in raw_submission_comments:
            if pd.notna(comment['body']):
                comment_text = self._clean_text(str(comment['body']))
                if comment_text and comment_text not in ['[removed]', '[deleted]']:
                    # Format comment with score
                    formatted_comment = self._format_comment_with_score(
                        comment_text,
                        comment.get('score', 0),
                        "Top level comment"
                    )
                    comments_list.append(formatted_comment)
        
        # Skip submissions with no valid content
        if not comments_list:
            logger.debug(f"Skipping submission {submission_id} - no valid content after cleaning")
            return None
        
        # Calculate comment statistics
        comment_stats = self._calculate_comment_stats(raw_submission_comments, comments_list)
        
        # Build final JSON structure
        json_data = {
            "submission_id": submission_id,
            "title": self._clean_text(str(submission_row['title'])),
            "date": formatted_date,
            "subreddit": subreddit,
            "comments": comments_list,
            "metadata": {
                "original_score": int(submission_row.get('score', 0)),
                "author": str(submission_row.get('author', '')),
                "original_link": str(submission_row['link']),
                "comment_count": len(raw_submission_comments),
                "valid_comment_count": len(comments_list),
                "conversion_source": "csv_dump",
                "url": str(submission_row.get('url', '')) if pd.notna(submission_row.get('url')) else None,
                "cleaning_applied": True,
                "year_filtering_applied": bool(self.config.min_year or self.config.max_year),
                "comment_breakdown": {
                    "top_level_comments": comment_stats.get('top_level', 0),
                    "total_replies": comment_stats.get('replies', 0),  # Will be 0 for CSV data
                    "empty_comments": comment_stats.get('empty', 0),
                    "threads_with_replies": 0  # CSV data doesn't have reply structure
                },
                "data_quality": {
                    "comments_with_scores": comment_stats.get('with_scores', 0),
                    "deleted_comments": comment_stats.get('deleted_removed', 0),
                    "processing_success_rate": round(
                        (len(raw_submission_comments) - comment_stats.get('empty', 0)) / max(len(raw_submission_comments), 1) * 100, 2
                    ) if raw_submission_comments else 100.0
                }
            }
        }
        
        return json_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text or text == 'nan' or text in ['[removed]', '[deleted]']:
            return ""
        
        # Decode HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#x200B;', '')  # Zero-width space
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _get_date_folder_path(self, created_date_str: str, subreddit: str) -> Path:
        """
        Get the folder path for organizing files by date
        
        Args:
            created_date_str: Date string from CSV
            subreddit: Subreddit name
            
        Returns:
            Path object for the date-organized folder
        """
        try:
            # Parse the date
            created_date = datetime.strptime(created_date_str, self.config.date_format)
            date_folder = created_date.strftime(self.config.folder_date_format)  # YYYYMMDD
            
            # Build path: output_directory/[subreddit/]YYYYMMDD/
            if self.config.organize_by_subreddit:
                folder_path = self.config.output_directory / subreddit / date_folder
            else:
                folder_path = self.config.output_directory / date_folder
            
            return folder_path
            
        except Exception as e:
            logger.error(f"Failed to parse date {created_date_str}: {e}")
            # Fallback to 'unknown_date' folder
            if self.config.organize_by_subreddit:
                return self.config.output_directory / subreddit / "unknown_date"
            else:
                return self.config.output_directory / "unknown_date"
    
    def _save_json_file_organized(self, json_data: Dict[str, Any], subreddit: str, created_date_str: str):
        """Save JSON data to organized folder structure"""
        
        # Get the appropriate folder path
        folder_path = self._get_date_folder_path(created_date_str, subreddit)
        
        # Create folder if it doesn't exist
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Track folder creation for stats
        folder_key = str(folder_path)
        if folder_key not in self.created_folders:
            self.created_folders.add(folder_key)
            self.stats['date_folders_created'] += 1
            logger.debug(f"Created folder: {folder_path}")
        
        # Create filename: submission_id.json (no subreddit prefix since it's in folder structure)
        filename = f"{json_data['submission_id']}.json"
        file_path = folder_path / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved: {file_path}")
            self.stats['files_created'] += 1
            
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
            self.stats['errors'] += 1
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get detailed conversion statistics"""
        return {
            'subreddits_processed': self.stats['subreddits_processed'],
            'submissions_processed': self.stats['submissions_processed'],
            'comments_processed': self.stats['comments_processed'],
            'files_created': self.stats['files_created'],
            'errors': self.stats['errors'],
            'submissions_removed': self.stats['submissions_removed'],
            'comments_removed': self.stats['comments_removed'],
            'orphan_comments_removed': self.stats['orphan_comments_removed'],
            'date_folders_created': self.stats['date_folders_created'],
            'date_parse_errors': self.stats['date_parse_errors'],
            'submissions_filtered_by_year': self.stats['submissions_filtered_by_year'],
            'comments_filtered_by_year': self.stats['comments_filtered_by_year'],
            'success_rate': (self.stats['files_created'] / max(self.stats['submissions_processed'], 1)) * 100,
            'data_quality': {
                'submission_removal_rate': (self.stats['submissions_removed'] / max(self.stats['submissions_processed'] + self.stats['submissions_removed'], 1)) * 100,
                'comment_removal_rate': (self.stats['comments_removed'] / max(self.stats['comments_processed'] + self.stats['comments_removed'], 1)) * 100,
                'orphan_removal_rate': (self.stats['orphan_comments_removed'] / max(self.stats['comments_processed'] + self.stats['comments_removed'] + self.stats['orphan_comments_removed'], 1)) * 100,
                'date_parse_error_rate': (self.stats['date_parse_errors'] / max(self.stats['submissions_processed'], 1)) * 100,
                'year_filtering_rate': {
                    'submissions': (self.stats['submissions_filtered_by_year'] / max(self.stats['submissions_filtered_by_year'] + self.stats['submissions_processed'], 1)) * 100,
                    'comments': (self.stats['comments_filtered_by_year'] / max(self.stats['comments_filtered_by_year'] + self.stats['comments_processed'], 1)) * 100
                }
            }
        }
    
    def print_folder_structure_sample(self):
        """Print a sample of the created folder structure"""
        logger.info("Sample folder structure created:")
        sample_folders = sorted(list(self.created_folders))[:10]  # Show first 10 folders
        for folder in sample_folders:
            logger.info(f"  {folder}")
        if len(self.created_folders) > 10:
            logger.info(f"  ... and {len(self.created_folders) - 10} more folders")