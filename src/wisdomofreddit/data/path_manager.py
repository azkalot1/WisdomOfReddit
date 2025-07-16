from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import logging
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    """Configuration for file storage"""
    base_directory: Path
    date_format: str = '%Y%m%d'  # YYYYMMDD
    file_extension: str = '.json'
    organize_by_subreddit: bool = True
    organize_by_date: bool = True

@dataclass
class S3StorageConfig:
    """Configuration for S3 storage"""
    bucket_name: str
    prefix: str = ""  # Base prefix in bucket (e.g., "reddit-data")
    date_format: str = '%Y%m%d'  # YYYYMMDD
    file_extension: str = '.json'
    organize_by_subreddit: bool = True
    organize_by_date: bool = True
    region: str = 'us-east-2'

class FilePathManager:
    """
    Manages file paths and directory structure for Reddit submissions
    
    Single Responsibility: File path generation and directory management
    No knowledge of Reddit API or JSON content - just path operations
    """
    
    def __init__(self, storage_config: StorageConfig):
        """
        Initialize path manager with storage configuration
        
        Args:
            storage_config: Configuration for file organization
        """
        self.config = storage_config
        self.base_path = Path(storage_config.base_directory)
    
    def get_submission_file_path(self, submission_id: str, created_date: Union[datetime, str], subreddit: str) -> Path:
        """
        Generate complete file path for a Reddit submission
        
        Args:
            submission_id: Reddit submission ID
            created_date: When the submission was created (datetime object or ISO string)
            subreddit: Subreddit name
            
        Returns:
            Complete file path: base_dir/YYYYMMDD/submission_id.json (if organize_by_subreddit=False)
                            or base_dir/subreddit/YYYYMMDD/submission_id.json (if organize_by_subreddit=True)
        """
        try:
            path_parts = [self.base_path]
            
            # Convert string date to datetime if needed
            if isinstance(created_date, str):
                try:
                    date_obj = datetime.fromisoformat(created_date)
                except ValueError as e:
                    logger.warning(f"Failed to parse date string '{created_date}': {e}. Using current time.")
                    date_obj = datetime.utcnow()
            else:
                date_obj = created_date
            
            # Add subreddit directory if configured
            if self.config.organize_by_subreddit:
                # Sanitize subreddit name for filesystem
                safe_subreddit = self._sanitize_filename(subreddit)
                path_parts.append(Path(safe_subreddit))
            
            # Add date directory if configured
            if self.config.organize_by_date:
                date_str = date_obj.strftime(self.config.date_format)  # ✅ Now using datetime object
                path_parts.append(Path(date_str))
            
            # Build directory path
            directory_path = Path(*path_parts)
            
            # Add filename
            safe_submission_id = self._sanitize_filename(submission_id)
            filename = f"{safe_submission_id}{self.config.file_extension}"
            
            file_path = directory_path / filename
            
            logger.debug(f"Generated file path: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to generate file path for {submission_id}: {e}")
            # Fallback to basic path
            return self.base_path / f"{submission_id}{self.config.file_extension}"
    
    def ensure_directory_exists(self, file_path: Path) -> bool:
        """
        Create directory structure if it doesn't exist
        
        Args:
            file_path: Complete file path (directory will be created for parent)
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            directory_path = file_path.parent
            directory_path.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Ensured directory exists: {directory_path}")
            return True
            
        except (OSError, IOError) as e:
            logger.error(f"Failed to create directory for {file_path}: {e}")
            return False
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to be filesystem-safe
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for all filesystems
        """
        # Replace problematic characters
        replacements = {
            '/': '_', '\\': '_', ':': '_', '*': '_', 
            '?': '_', '"': '_', '<': '_', '>': '_', '|': '_',
            '\n': '_', '\r': '_', '\t': '_'
        }
        
        sanitized = filename
        for char, replacement in replacements.items():
            sanitized = sanitized.replace(char, replacement)
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'unnamed'
        
        # Limit length to prevent filesystem issues
        max_length = 200
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def get_directory_info(self, directory_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Get information about a directory
        
        Args:
            directory_path: Directory to analyze (default: base directory)
            
        Returns:
            Dictionary with directory information
        """
        if directory_path is None:
            directory_path = self.base_path
        
        info = {
            'path': str(directory_path),
            'exists': False,
            'is_directory': False,
            'file_count': 0,
            'subdirectory_count': 0,
            'total_size_bytes': 0,
            'json_files': 0
        }
        
        try:
            if directory_path.exists():
                info['exists'] = True
                info['is_directory'] = directory_path.is_dir()
                
                if info['is_directory']:
                    files = list(directory_path.iterdir())
                    
                    for item in files:
                        if item.is_file():
                            info['file_count'] += 1
                            info['total_size_bytes'] += item.stat().st_size
                            
                            if item.suffix.lower() == '.json':
                                info['json_files'] += 1
                        
                        elif item.is_dir():
                            info['subdirectory_count'] += 1
            
        except (OSError, IOError) as e:
            logger.error(f"Failed to get directory info for {directory_path}: {e}")
        
        return info
    
    def list_submission_files(self, subreddit: Optional[str] = None, 
                            date_filter: Optional[datetime] = None) -> List[Path]:
        """
        List all submission files, optionally filtered
        
        Args:
            subreddit: Filter by subreddit (None for all)
            date_filter: Filter by date (None for all)
            
        Returns:
            List of file paths matching filters
        """
        files = []
        
        try:
            search_path = self.base_path
            
            # Navigate to subreddit if specified
            if subreddit and self.config.organize_by_subreddit:
                search_path = search_path / self._sanitize_filename(subreddit)
                if not search_path.exists():
                    return files
            
            # Navigate to date if specified
            if date_filter and self.config.organize_by_date:
                date_str = date_filter.strftime(self.config.date_format)
                search_path = search_path / date_str
                if not search_path.exists():
                    return files
            
            # Find JSON files
            if search_path.exists():
                pattern = f"*{self.config.file_extension}"
                if self.config.organize_by_date or self.config.organize_by_subreddit:
                    # Search recursively if organized
                    files = list(search_path.rglob(pattern))
                else:
                    # Search in single directory
                    files = list(search_path.glob(pattern))
            
            logger.info(f"Found {len(files)} submission files")
            
        except Exception as e:
            logger.error(f"Failed to list submission files: {e}")
        
        return sorted(files)
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'base_directory': str(self.base_path),
            'total_files': 0,
            'total_size_bytes': 0,
            'subreddits': {},
            'dates': {},
            'config': {
                'organize_by_subreddit': self.config.organize_by_subreddit,
                'organize_by_date': self.config.organize_by_date,
                'file_extension': self.config.file_extension
            }
        }
        
        try:
            all_files = self.list_submission_files()
            stats['total_files'] = len(all_files)
            
            for file_path in all_files:
                try:
                    # Add to total size
                    stats['total_size_bytes'] += file_path.stat().st_size
                    
                    # Extract subreddit and date from path
                    parts = file_path.relative_to(self.base_path).parts
                    
                    if self.config.organize_by_subreddit and len(parts) > 0:
                        subreddit = parts[0]
                        if subreddit not in stats['subreddits']:
                            stats['subreddits'][subreddit] = 0
                        stats['subreddits'][subreddit] += 1
                    
                    if self.config.organize_by_date and len(parts) > 1:
                        date_part = parts[1] if self.config.organize_by_subreddit else parts[0]
                        if date_part not in stats['dates']:
                            stats['dates'][date_part] = 0
                        stats['dates'][date_part] += 1
                
                except (OSError, IOError):
                    continue
        
        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
        
        return stats


class S3PathManager:
    """
    Manages S3 paths and key structure for Reddit submissions
    
    Implements same interface as FilePathManager but for S3
    """
    
    def __init__(self, storage_config: S3StorageConfig):
        """
        Initialize S3 path manager with storage configuration
        
        Args:
            storage_config: Configuration for S3 organization
        """
        self.config = storage_config
        self.bucket_name = storage_config.bucket_name
        self.base_prefix = storage_config.prefix.rstrip('/') if storage_config.prefix else ""
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=storage_config.region)
        
        # Verify bucket exists
        self._verify_bucket_exists()
    
    def _verify_bucket_exists(self):
        """Verify that the S3 bucket exists and is accessible"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Verified access to S3 bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket does not exist: {self.bucket_name}")
            else:
                logger.error(f"Cannot access S3 bucket {self.bucket_name}: {e}")
            raise
    
    def get_submission_file_path(self, submission_id: str, created_date: Union[datetime, str], subreddit: str) -> Path:
        """
        Generate complete file path for a Reddit submission
        
        IMPORTANT: Returns a Path object for compatibility, but it represents an S3 key
        
        Args:
            submission_id: Reddit submission ID
            created_date: When the submission was created (datetime object or ISO string)
            subreddit: Subreddit name
            
        Returns:
            Path object representing S3 key (for interface compatibility)
        """
        s3_key = self._generate_s3_key(submission_id, created_date, subreddit)
        # Return as Path for interface compatibility
        return Path(s3_key)
    
    def _generate_s3_key(self, submission_id: str, created_date: Union[datetime, str], subreddit: str) -> str:
        """
        Generate S3 key for a Reddit submission
        
        Args:
            submission_id: Reddit submission ID
            created_date: When the submission was created
            subreddit: Subreddit name
            
        Returns:
            S3 key string
        """
        try:
            key_parts = []
            
            # Add base prefix if configured
            if self.base_prefix:
                key_parts.append(self.base_prefix)
            
            # Convert string date to datetime if needed
            if isinstance(created_date, str):
                try:
                    date_obj = datetime.fromisoformat(created_date)
                except ValueError as e:
                    logger.warning(f"Failed to parse date string '{created_date}': {e}. Using current time.")
                    date_obj = datetime.utcnow()
            else:
                date_obj = created_date
            
            # Add subreddit directory if configured
            if self.config.organize_by_subreddit:
                # Sanitize subreddit name for S3
                safe_subreddit = self._sanitize_filename(subreddit)
                key_parts.append(safe_subreddit)
            
            # Add date directory if configured
            if self.config.organize_by_date:
                date_str = date_obj.strftime(self.config.date_format)
                key_parts.append(date_str)
            
            # Add filename
            safe_submission_id = self._sanitize_filename(submission_id)
            filename = f"{safe_submission_id}{self.config.file_extension}"
            key_parts.append(filename)
            
            # Join with forward slashes
            s3_key = '/'.join(key_parts)
            
            logger.debug(f"Generated S3 key: {s3_key}")
            return s3_key
            
        except Exception as e:
            logger.error(f"Failed to generate S3 key for {submission_id}: {e}")
            # Fallback to basic key
            if self.base_prefix:
                return f"{self.base_prefix}/{submission_id}{self.config.file_extension}"
            return f"{submission_id}{self.config.file_extension}"
    
    def ensure_directory_exists(self, file_path: Path) -> bool:
        """
        S3 doesn't need directory creation, but we keep this method for compatibility
        
        Args:
            file_path: Complete file path (ignored for S3)
            
        Returns:
            Always True for S3
        """
        # S3 doesn't have directories, so this is a no-op
        logger.debug(f"ensure_directory_exists called for {file_path} (no-op for S3)")
        return True
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to be S3-key safe
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for S3 keys
        """
        # S3 is more permissive than filesystems, but we still want clean keys
        replacements = {
            '/': '_', '\\': '_', ':': '_', '*': '_', 
            '?': '_', '"': '_', '<': '_', '>': '_', '|': '_',
            '\n': '_', '\r': '_', '\t': '_', ' ': '_',
            '#': '_'
        }
        
        sanitized = filename
        for char, replacement in replacements.items():
            sanitized = sanitized.replace(char, replacement)
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'unnamed'
        
        # Limit length
        max_length = 200
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def list_submission_files(self, subreddit: Optional[str] = None, 
                            date_filter: Optional[datetime] = None) -> List[Path]:
        """
        List all submission files, optionally filtered
        
        Args:
            subreddit: Filter by subreddit (None for all)
            date_filter: Filter by date (None for all)
            
        Returns:
            List of Path objects (representing S3 keys)
        """
        keys = []
        
        try:
            # Build prefix for listing
            prefix_parts = []
            if self.base_prefix:
                prefix_parts.append(self.base_prefix)
            
            if subreddit and self.config.organize_by_subreddit:
                prefix_parts.append(self._sanitize_filename(subreddit))
            
            if date_filter and self.config.organize_by_date:
                date_str = date_filter.strftime(self.config.date_format)
                prefix_parts.append(date_str)
            
            prefix = '/'.join(prefix_parts) if prefix_parts else ''
            
            # List objects with prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith(self.config.file_extension):
                            keys.append(Path(key))  # Return as Path for compatibility
            
            logger.info(f"Found {len(keys)} submission files in S3")
            
        except ClientError as e:
            logger.error(f"Failed to list S3 objects: {e}")
        
        return sorted(keys)
    
    def get_directory_info(self, directory_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Get information about a directory (S3 prefix)
        
        Args:
            directory_path: Directory to analyze (default: base prefix)
            
        Returns:
            Dictionary with directory information
        """
        if directory_path is None:
            prefix = self.base_prefix
        else:
            prefix = str(directory_path)
        
        info = {
            'path': prefix,
            'exists': True,  # S3 prefixes always "exist"
            'is_directory': True,
            'file_count': 0,
            'subdirectory_count': 0,
            'total_size_bytes': 0,
            'json_files': 0
        }
        
        try:
            # List objects with prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix,
                Delimiter='/'
            )
            
            subdirs = set()
            
            for page in page_iterator:
                # Count files
                if 'Contents' in page:
                    for obj in page['Contents']:
                        info['file_count'] += 1
                        info['total_size_bytes'] += obj['Size']
                        
                        if obj['Key'].endswith('.json'):
                            info['json_files'] += 1
                
                # Count subdirectories
                if 'CommonPrefixes' in page:
                    for prefix_info in page['CommonPrefixes']:
                        subdirs.add(prefix_info['Prefix'])
            
            info['subdirectory_count'] = len(subdirs)
            
        except ClientError as e:
            logger.error(f"Failed to get directory info for {prefix}: {e}")
        
        return info
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'bucket_name': self.bucket_name,
            'base_prefix': self.base_prefix,
            'total_files': 0,
            'total_size_bytes': 0,
            'subreddits': {},
            'dates': {},
            'config': {
                'organize_by_subreddit': self.config.organize_by_subreddit,
                'organize_by_date': self.config.organize_by_date,
                'file_extension': self.config.file_extension
            }
        }
        
        try:
            # List all objects
            paginator = self.s3_client.get_paginator('list_objects_v2')
            prefix = self.base_prefix if self.base_prefix else ''
            
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if not key.endswith(self.config.file_extension):
                            continue
                        
                        stats['total_files'] += 1
                        stats['total_size_bytes'] += obj['Size']
                        
                        # Extract subreddit and date from key
                        if self.base_prefix:
                            relative_key = key[len(self.base_prefix):].lstrip('/')
                        else:
                            relative_key = key
                        
                        parts = relative_key.split('/')
                        
                        if self.config.organize_by_subreddit and len(parts) > 0:
                            subreddit = parts[0]
                            if subreddit not in stats['subreddits']:
                                stats['subreddits'][subreddit] = 0
                            stats['subreddits'][subreddit] += 1
                        
                        if self.config.organize_by_date:
                            date_idx = 1 if self.config.organize_by_subreddit else 0
                            if len(parts) > date_idx:
                                date_part = parts[date_idx]
                                if date_part not in stats['dates']:
                                    stats['dates'][date_part] = 0
                                stats['dates'][date_part] += 1
        
        except ClientError as e:
            logger.error(f"Failed to get S3 storage statistics: {e}")
        
        return stats