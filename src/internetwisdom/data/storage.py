import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import logging
from abc import ABC, abstractmethod
import boto3
from enum import Enum
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

logger = logging.getLogger(__name__)


class ExistsStrategy(Enum):
    """Strategy for handling exists() checks"""
    CACHE = "cache"
    DIRECT = "direct"
    MOCK_FALSE = "mock_false"
    MOCK_TRUE = "mock_true"

class DataStorage(ABC):
    """
    Abstract base class for data persistence with shared functionality
    
    Provides common methods that work across different storage backends
    """
    
    @abstractmethod
    def save(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save data to storage"""
        pass
    
    @abstractmethod
    def load(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load data from storage"""
        pass
    
    @abstractmethod
    def exists(self, file_path: Path) -> bool:
        """Check if file exists in storage"""
        pass
    
    @abstractmethod
    def get_file_size(self, file_path: Path) -> Optional[int]:
        """Get file size in bytes"""
        pass
    
    # Shared methods that work for any storage backend
    def backup_file(self, file_path: Path, backup_suffix: str = '.backup') -> bool:
        """
        Create a backup copy of existing file
        
        This method works for any storage backend since it uses the abstract methods.
        
        Args:
            file_path: Original file path
            backup_suffix: Suffix to add to backup file
            
        Returns:
            True if backup created successfully
        """
        try:
            if not self.exists(file_path):
                logger.warning(f"Cannot backup non-existent file: {file_path}")
                return False
            
            # Create backup path
            backup_path = self._create_backup_path(file_path, backup_suffix)
            
            # Load original data
            original_data = self.load(file_path)
            if original_data is None:
                logger.error(f"Failed to load original data for backup: {file_path}")
                return False
            
            # Save to backup location
            success = self.save(original_data, backup_path)
            if success:
                logger.info(f"Created backup: {file_path} -> {backup_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to backup file {file_path}: {e}")
            return False
    
    def _create_backup_path(self, file_path: Path, backup_suffix: str) -> Path:
        """
        Create backup path from original path
        
        Can be overridden by subclasses for custom backup naming
        """
        if hasattr(file_path, 'suffix'):
            # For local files
            return file_path.with_suffix(file_path.suffix + backup_suffix)
        else:
            # For S3 keys (represented as Path)
            return Path(str(file_path) + backup_suffix)
    
    def validate_json_structure(self, file_path: Path, required_fields: list = None) -> Dict[str, Any]:
        """
        Validate JSON file structure
        
        This method works for any storage backend since it uses the abstract methods.
        
        Args:
            file_path: Path to JSON file
            required_fields: List of required field names
            
        Returns:
            Validation results dictionary
        """
        validation_result = {
            'is_valid': False,
            'exists': False,
            'is_readable': False,
            'has_required_fields': False,
            'missing_fields': [],
            'file_size': None,
            'issues': []
        }
        
        try:
            # Check if file exists
            validation_result['exists'] = self.exists(file_path)
            if not validation_result['exists']:
                validation_result['issues'].append('File does not exist')
                return validation_result
            
            # Check file size
            validation_result['file_size'] = self.get_file_size(file_path)
            
            # Try to load JSON
            data = self.load(file_path)
            if data is None:
                validation_result['issues'].append('Cannot read or parse JSON')
                return validation_result
            
            validation_result['is_readable'] = True
            
            # Check required fields if specified
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                validation_result['missing_fields'] = missing_fields
                validation_result['has_required_fields'] = len(missing_fields) == 0
                
                if missing_fields:
                    validation_result['issues'].append(f'Missing required fields: {missing_fields}')
            else:
                validation_result['has_required_fields'] = True
            
            # Overall validation
            validation_result['is_valid'] = (
                validation_result['exists'] and 
                validation_result['is_readable'] and 
                validation_result['has_required_fields']
            )
            
        except Exception as e:
            validation_result['issues'].append(f'Validation error: {str(e)}')
            logger.error(f"JSON validation failed for {file_path}: {e}")
        
        return validation_result
    
    def copy_file(self, source_path: Path, dest_path: Path) -> bool:
        """
        Copy file from source to destination
        
        Works across any storage backend using load/save
        """
        try:
            if not self.exists(source_path):
                logger.error(f"Source file does not exist: {source_path}")
                return False
            
            # Load from source
            data = self.load(source_path)
            if data is None:
                logger.error(f"Failed to load source file: {source_path}")
                return False
            
            # Save to destination
            success = self.save(data, dest_path)
            if success:
                logger.info(f"Copied file: {source_path} -> {dest_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to copy file {source_path} -> {dest_path}: {e}")
            return False
    
    def get_storage_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get comprehensive information about a file
        
        Returns:
            Dictionary with file information
        """
        info = {
            'path': str(file_path),
            'exists': False,
            'size_bytes': None,
            'is_valid_json': False,
            'validation_issues': []
        }
        
        try:
            info['exists'] = self.exists(file_path)
            
            if info['exists']:
                info['size_bytes'] = self.get_file_size(file_path)
                
                # Quick JSON validation
                validation = self.validate_json_structure(file_path)
                info['is_valid_json'] = validation['is_valid']
                info['validation_issues'] = validation['issues']
        
        except Exception as e:
            info['validation_issues'].append(f'Error getting file info: {str(e)}')
        
        return info

class JsonFileStorage(DataStorage):
    """
    Handles JSON file operations for local filesystem
    
    Single Responsibility: Pure JSON file I/O operations
    No knowledge of Reddit data or file path generation
    """
    
    def __init__(self, encoding: str = 'utf-8', indent: int = 2):
        """
        Initialize JSON file storage
        
        Args:
            encoding: File encoding for JSON files
            indent: JSON indentation for pretty printing
        """
        self.encoding = encoding
        self.indent = indent
    
    def save(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save data to JSON file"""
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON file
            with open(file_path, 'w', encoding=self.encoding) as f:
                json.dump(data, f, indent=self.indent, ensure_ascii=False)
            
            logger.info(f"Successfully saved JSON to {file_path}")
            return True
            
        except (IOError, OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    def load(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            if not file_path.exists():
                logger.warning(f"JSON file does not exist: {file_path}")
                return None
            
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            logger.debug(f"Successfully loaded JSON from {file_path}")
            return data
            
        except (IOError, OSError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None
    
    def exists(self, file_path: Path) -> bool:
        """Check if JSON file exists"""
        try:
            return file_path.exists() and file_path.is_file()
        except (OSError, IOError):
            return False
    
    def get_file_size(self, file_path: Path) -> Optional[int]:
        """Get file size in bytes"""
        try:
            if self.exists(file_path):
                return file_path.stat().st_size
            return None
        except (OSError, IOError) as e:
            logger.error(f"Failed to get file size for {file_path}: {e}")
            return None

class S3JsonStorage(DataStorage):
    """
    Handles S3 JSON file operations with configurable exists() behavior
    
    Implements same interface as JsonFileStorage but for S3
    """
    
    def __init__(self, bucket_name: str, region: str = 'us-east-2', 
                 encoding: str = 'utf-8', indent: int = 2,
                 exists_strategy: ExistsStrategy = ExistsStrategy.MOCK_FALSE):
        """
        Initialize S3 JSON storage
        
        Args:
            bucket_name: S3 bucket name
            region: AWS region
            encoding: File encoding for JSON files
            indent: JSON indentation for pretty printing
            exists_strategy: Strategy for handling exists() checks
        """
        self.bucket_name = bucket_name
        self.encoding = encoding
        self.indent = indent
        self.exists_strategy = exists_strategy
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Cache for exists() checks (only used if strategy is CACHE)
        self._exists_cache: Set[str] = set()
        self._cache_lock = Lock()
        self._cache_initialized = False
        
        # Verify bucket access
        self._verify_bucket_access()
        
        # Initialize cache if using cache strategy
        if self.exists_strategy == ExistsStrategy.CACHE:
            self._initialize_cache()
        
        logger.info(f"Initialized S3JsonStorage with exists_strategy: {exists_strategy.value}")
    
    def _verify_bucket_access(self):
        """Verify we can access the S3 bucket"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            logger.error(f"Cannot access S3 bucket {self.bucket_name}: {e}")
            raise
    
    def _initialize_cache(self):
        """Initialize the exists cache by listing all current objects"""
        if self._cache_initialized:
            return
        
        logger.info("Initializing S3 exists cache...")
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name)
            
            count = 0
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        self._exists_cache.add(obj['Key'])
                        count += 1
            
            self._cache_initialized = True
            logger.info(f"Cached {count} existing S3 objects")
        except ClientError as e:
            logger.error(f"Failed to initialize cache: {e}")
            # Fall back to direct strategy
            self.exists_strategy = ExistsStrategy.DIRECT
    
    def save(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save data to S3 as JSON"""
        # Convert Path to S3 key
        s3_key = str(file_path)
        
        try:
            # Convert data to JSON
            json_data = json.dumps(data, indent=self.indent, ensure_ascii=False)
            json_bytes = json_data.encode(self.encoding)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_bytes,
                ContentType='application/json',
                ContentEncoding=self.encoding
            )
            
            # Update cache if using cache strategy
            if self.exists_strategy == ExistsStrategy.CACHE:
                with self._cache_lock:
                    self._exists_cache.add(s3_key)
            
            logger.info(f"Successfully saved JSON to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except (ClientError, TypeError, ValueError) as e:
            logger.error(f"Failed to save JSON to S3 key {s3_key}: {e}")
            return False
    
    def load(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load data from S3 JSON file"""
        # Convert Path to S3 key
        s3_key = str(file_path)
        
        try:
            # Get object from S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            # Read and parse JSON
            json_content = response['Body'].read().decode(self.encoding)
            data = json.loads(json_content)
            
            logger.debug(f"Successfully loaded JSON from s3://{self.bucket_name}/{s3_key}")
            return data
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"S3 object does not exist: {s3_key}")
            else:
                logger.error(f"Failed to load JSON from S3: {e}")
            return None
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from {s3_key}: {e}")
            return None
    
    def exists(self, file_path: Path) -> bool:
        """Check if S3 object exists based on configured strategy"""
        # Convert Path to S3 key
        s3_key = str(file_path)
        
        # Handle different strategies
        if self.exists_strategy == ExistsStrategy.MOCK_FALSE:
            logger.debug(f"exists() mocked to False for {s3_key}")
            return False
        
        elif self.exists_strategy == ExistsStrategy.MOCK_TRUE:
            logger.debug(f"exists() mocked to True for {s3_key}")
            return True
        
        elif self.exists_strategy == ExistsStrategy.CACHE:
            # Ensure cache is initialized
            if not self._cache_initialized:
                self._initialize_cache()
            
            with self._cache_lock:
                exists = s3_key in self._exists_cache
            
            logger.debug(f"exists() cache check for {s3_key}: {exists}")
            return exists
        
        elif self.exists_strategy == ExistsStrategy.DIRECT:
            # Check S3 directly
            try:
                self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                logger.debug(f"exists() direct check for {s3_key}: True")
                return True
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.debug(f"exists() direct check for {s3_key}: False")
                    return False
                else:
                    logger.error(f"Error checking S3 object existence: {e}")
                    return False
        
        else:
            logger.error(f"Unknown exists strategy: {self.exists_strategy}")
            return False
    
    def get_file_size(self, file_path: Path) -> Optional[int]:
        """Get S3 object size in bytes"""
        # Convert Path to S3 key
        s3_key = str(file_path)
        
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response['ContentLength']
        except ClientError as e:
            logger.error(f"Failed to get S3 object size for {s3_key}: {e}")
            return None
    
    # S3-specific methods
    def set_exists_strategy(self, strategy: ExistsStrategy):
        """Change the exists strategy at runtime"""
        old_strategy = self.exists_strategy
        self.exists_strategy = strategy
        
        logger.info(f"Changed exists strategy from {old_strategy.value} to {strategy.value}")
        
        # Initialize cache if switching to cache strategy
        if strategy == ExistsStrategy.CACHE and not self._cache_initialized:
            self._initialize_cache()
    
    def get_exists_stats(self) -> Dict[str, Any]:
        """Get statistics about exists() behavior"""
        stats = {
            'strategy': self.exists_strategy.value,
            'cache_initialized': self._cache_initialized,
            'cached_objects': len(self._exists_cache) if self.exists_strategy == ExistsStrategy.CACHE else 0
        }
        
        return stats
    
    def refresh_cache(self):
        """Refresh the exists cache (only relevant for CACHE strategy)"""
        if self.exists_strategy == ExistsStrategy.CACHE:
            logger.info("Refreshing S3 exists cache...")
            with self._cache_lock:
                self._exists_cache.clear()
                self._cache_initialized = False
            self._initialize_cache()
        else:
            logger.warning(f"refresh_cache() called but strategy is {self.exists_strategy.value}")
    
    def _create_backup_path(self, file_path: Path, backup_suffix: str) -> Path:
        """Override backup path creation for S3 keys"""
        return Path(str(file_path) + backup_suffix)