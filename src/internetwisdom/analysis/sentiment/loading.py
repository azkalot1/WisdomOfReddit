from typing import Protocol, Optional, List
from .state import RedditContent
from pathlib import Path   
import logging
logger = logging.getLogger(__name__)
import json
from .utils import create_reddit_content

class ContentLoader(Protocol):
    """Interface for loading Reddit content"""
    
    def load_day_content(self, date: str) -> List[RedditContent]:
        """Load all Reddit content for a specific day"""
        ...
    
    def load_submission_content(self, date: str, submission_id: str) -> Optional[RedditContent]:
        """Load content for a specific submission"""
        ...
    
    def get_available_dates(self) -> List[str]:
        """Get list of available dates"""
        ...
    
    def get_submissions_for_date(self, date: str) -> List[str]:
        """Get list of submission IDs for a date"""
        ...

class LocalFileContentLoader:
    """Load Reddit content from local JSON files"""
    
    def __init__(self, base_path: Path, max_n_comments: Optional[int] = None):
        self.base_path = Path(base_path)
        self.max_n_comments = max_n_comments
    
    def load_day_content(self, date: str) -> List[RedditContent]:
        """Load all Reddit content for a specific day"""
        date_path = self.base_path / date
        if not date_path.exists():
            logger.warning(f"Date folder not found: {date_path}")
            return []
        
        contents = []
        for json_file in date_path.glob("*.json"):
            try:
                content = self.load_submission_content(date, json_file.stem)
                if content:
                    contents.append(content)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(contents)} submissions for {date}")
        return contents
    
    def load_submission_content(self, date: str, submission_id: str) -> Optional[RedditContent]:
        """Load content for a specific submission"""
        file_path = self.base_path / date / f"{submission_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return create_reddit_content(data, self.max_n_comments)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def get_available_dates(self) -> List[str]:
        """Get list of available dates"""
        dates = []
        for folder in self.base_path.iterdir():
            if folder.is_dir() and folder.name.isdigit() and len(folder.name) == 8:
                dates.append(folder.name)
        return sorted(dates)
    
    def get_submissions_for_date(self, date: str) -> List[str]:
        """Get list of submission IDs for a date"""
        date_path = self.base_path / date
        if not date_path.exists():
            return []
        
        return [f.stem for f in date_path.glob("*.json")]


class S3ContentLoader:
    """Load Reddit content from S3 storage"""
    
    def __init__(self, bucket: str, prefix: str, s3_client=None):
        self.bucket = bucket
        self.prefix = prefix
        self.s3_client = s3_client or self._create_s3_client()
    
    def _create_s3_client(self):
        import boto3
        return boto3.client('s3')
    
    def load_day_content(self, date: str) -> List[RedditContent]:
        """Load all Reddit content for a specific day from S3"""
        # Implementation for S3 loading
        # This would list objects with prefix f"{self.prefix}/{date}/"
        # and load each JSON file
        pass  # Implement based on your S3 structure
    
    def load_submission_content(self, date: str, submission_id: str) -> Optional[RedditContent]:
        """Load content for a specific submission from S3"""
        key = f"{self.prefix}/{date}/{submission_id}.json"
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            data = json.loads(response['Body'].read())
            return create_reddit_content(data)
        except Exception as e:
            logger.error(f"Failed to load s3://{self.bucket}/{key}: {e}")
            return None
    
    def get_available_dates(self) -> List[str]:
        """Get list of available dates from S3"""
        # Implementation to list date prefixes
        pass
    
    def get_submissions_for_date(self, date: str) -> List[str]:
        """Get list of submission IDs for a date from S3"""
        # Implementation to list submissions for a date
        pass