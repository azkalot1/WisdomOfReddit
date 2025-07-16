import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from .relevance_pipeline import RelevancePredictor

@dataclass
class PredictionResult:
    """Single prediction result"""
    submission_id: str
    date_folder: str
    predicted_label: int
    predicted_class: str
    probability_relevant: float
    confidence: float
    processing_time: float
    timestamp: datetime
    error: Optional[str] = None

@dataclass
class BatchPredictionResults:
    """Results from batch prediction"""
    total_processed: int
    successful_predictions: int
    failed_predictions: int
    relevant_count: int
    avg_confidence: float
    processing_time: float
    date_range: Tuple[str, str]
    model_info: Dict[str, Any]
    timestamp: datetime

class SubmissionBatchPredictor:
    """
    Efficiently predict relevance for large numbers of submissions
    """
    
    def __init__(self, 
                 predictor: 'RelevancePredictor',
                 base_data_path: Path,
                 batch_size: int = 1000,
                 num_workers: int = 4,
                 min_confidence_threshold: float = 0.7):
        """
        Initialize batch predictor
        
        Args:
            predictor: Trained RelevancePredictor
            base_data_path: Path to reddit data folder
            batch_size: Number of submissions to process in each batch
            num_workers: Number of parallel workers
            min_confidence_threshold: Minimum confidence for "relevant" classification
        """
        self.predictor = predictor
        self.base_data_path = Path(base_data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_confidence_threshold = min_confidence_threshold
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def discover_all_submissions(self) -> List[Tuple[str, str]]:
        """
        Discover all submission files in the data directory
        
        Returns:
            List of (date_folder, submission_id) tuples
        """
        self.logger.info(f"Discovering submissions in {self.base_data_path}")
        
        submissions = []
        pattern = str(self.base_data_path / "*" / "*.json")
        
        for file_path in glob.glob(pattern):
            path_obj = Path(file_path)
            date_folder = path_obj.parent.name
            submission_id = path_obj.stem
            submissions.append((date_folder, submission_id))
        
        self.logger.info(f"Found {len(submissions)} submissions across date folders")
        return submissions
    
    def load_submission_batch(self, submission_refs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Load a batch of submissions from disk
        
        Args:
            submission_refs: List of (date_folder, submission_id) tuples
            
        Returns:
            List of loaded submission data
        """
        submissions = []
        
        for date_folder, submission_id in submission_refs:
            try:
                file_path = self.base_data_path / date_folder / f"{submission_id}.json"
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Ensure submission_id is in the data
                data['submission_id'] = submission_id
                data['date_folder'] = date_folder
                submissions.append(data)
                
            except Exception as e:
                self.logger.warning(f"Failed to load {date_folder}/{submission_id}: {e}")
        
        return submissions
    
    def predict_batch(self, submissions: List[Dict[str, Any]]) -> List[PredictionResult]:
        """
        Predict relevance for a batch of submissions
        
        Args:
            submissions: List of submission data
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        start_time = datetime.now()
        
        # Use the predictor's batch prediction
        predictions = self.predictor.predict_batch(submissions)
        
        for submission, prediction in zip(submissions, predictions):
            processing_time = (datetime.now() - start_time).total_seconds() / len(submissions)
            
            if 'error' in prediction:
                result = PredictionResult(
                    submission_id=submission.get('submission_id', 'unknown'),
                    date_folder=submission.get('date_folder', 'unknown'),
                    predicted_label=0,
                    predicted_class='error',
                    probability_relevant=0.0,
                    confidence=0.0,
                    processing_time=processing_time,
                    timestamp=datetime.now(),
                    error=prediction['error']
                )
            else:
                result = PredictionResult(
                    submission_id=prediction['submission_id'],
                    date_folder=submission.get('date_folder', 'unknown'),
                    predicted_label=prediction['predicted_label'],
                    predicted_class=prediction['predicted_class'],
                    probability_relevant=prediction['probability_relevant'],
                    confidence=prediction['confidence'],
                    processing_time=processing_time,
                    timestamp=datetime.now()
                )
            
            results.append(result)
        
        return results
    
    def process_all_submissions(self, 
                              output_format: str = "parquet",  # "parquet", "csv", "sqlite", "json"
                              output_path: Path = None,
                              date_filter: Optional[Tuple[str, str]] = None,
                              resume_from: Optional[str] = None) -> BatchPredictionResults:
        """
        Process all submissions and save results
        
        Args:
            output_format: Format to save results ("parquet", "csv", "sqlite", "json")
            output_path: Path to save results
            date_filter: Optional (start_date, end_date) filter (format: "YYYYMMDD")
            resume_from: Resume processing from this date folder
            
        Returns:
            BatchPredictionResults summary
        """
        start_time = datetime.now()
        
        # Discover submissions
        all_submissions = self.discover_all_submissions()
        
        # Apply date filter if specified
        if date_filter:
            start_date, end_date = date_filter
            all_submissions = [
                (date_folder, sub_id) for date_folder, sub_id in all_submissions
                if start_date <= date_folder <= end_date
            ]
            self.logger.info(f"Filtered to {len(all_submissions)} submissions in date range {start_date}-{end_date}")
        
        # Resume functionality
        if resume_from:
            all_submissions = [
                (date_folder, sub_id) for date_folder, sub_id in all_submissions
                if date_folder >= resume_from
            ]
            self.logger.info(f"Resuming from {resume_from}: {len(all_submissions)} submissions remaining")
        
        # Setup output
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"prediction_results_{timestamp}")
        
        output_path = Path(output_path)
        
        # Initialize storage
        storage = self._initialize_storage(output_format, output_path)
        
        # Process in batches
        total_processed = 0
        successful_predictions = 0
        failed_predictions = 0
        relevant_count = 0
        confidence_sum = 0.0
        
        self.logger.info(f"Starting batch processing of {len(all_submissions)} submissions...")
        
        for i in range(0, len(all_submissions), self.batch_size):
            batch_refs = all_submissions[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(all_submissions) + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_refs)} submissions)")
            
            # Load batch
            submissions = self.load_submission_batch(batch_refs)
            
            if not submissions:
                self.logger.warning(f"No submissions loaded for batch {batch_num}")
                continue
            
            # Predict batch
            results = self.predict_batch(submissions)
            
            # Update statistics
            batch_successful = len([r for r in results if not r.error])
            batch_failed = len([r for r in results if r.error])
            batch_relevant = len([r for r in results if r.predicted_label == 1 and not r.error])
            batch_confidence_sum = sum(r.confidence for r in results if not r.error)
            
            total_processed += len(results)
            successful_predictions += batch_successful
            failed_predictions += batch_failed
            relevant_count += batch_relevant
            confidence_sum += batch_confidence_sum
            
            # Save batch results
            self._save_batch_results(storage, results, output_format)
            
            # Progress update
            self.logger.info(f"  Batch {batch_num} complete: {batch_successful} successful, "
                           f"{batch_relevant} relevant, {batch_failed} failed")
        
        # Finalize storage
        self._finalize_storage(storage, output_format, output_path)
        
        # Calculate final statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        avg_confidence = confidence_sum / successful_predictions if successful_predictions > 0 else 0.0
        
        # Determine date range
        if all_submissions:
            date_folders = [date_folder for date_folder, _ in all_submissions]
            date_range = (min(date_folders), max(date_folders))
        else:
            date_range = ("", "")
        
        results_summary = BatchPredictionResults(
            total_processed=total_processed,
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            relevant_count=relevant_count,
            avg_confidence=avg_confidence,
            processing_time=processing_time,
            date_range=date_range,
            model_info=self._get_model_info(),
            timestamp=datetime.now()
        )
        
        # Save summary
        self._save_summary(results_summary, output_path)
        
        self.logger.info(f"✅ Processing complete!")
        self.logger.info(f"  Total processed: {total_processed}")
        self.logger.info(f"  Successful: {successful_predictions}")
        self.logger.info(f"  Failed: {failed_predictions}")
        self.logger.info(f"  Relevant: {relevant_count} ({relevant_count/successful_predictions*100:.1f}%)")
        self.logger.info(f"  Avg confidence: {avg_confidence:.3f}")
        self.logger.info(f"  Processing time: {processing_time:.1f}s")
        self.logger.info(f"  Results saved to: {output_path}")
        
        return results_summary
    
    def _initialize_storage(self, output_format: str, output_path: Path) -> Any:
        """Initialize storage based on output format"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        if output_format == "sqlite":
            db_path = output_path / "predictions.db"
            conn = sqlite3.connect(str(db_path))
            
            # Create table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    submission_id TEXT,
                    date_folder TEXT,
                    predicted_label INTEGER,
                    predicted_class TEXT,
                    probability_relevant REAL,
                    confidence REAL,
                    processing_time REAL,
                    timestamp TEXT,
                    error TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_submission_id ON predictions(submission_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date_folder ON predictions(date_folder)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predicted_label ON predictions(predicted_label)")
            
            return conn
        
        elif output_format in ["parquet", "csv"]:
            # For parquet/csv, we'll collect results and save at the end
            return []
        
        elif output_format == "json":
            return []
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _save_batch_results(self, storage: Any, results: List[PredictionResult], output_format: str):
        """Save batch results to storage"""
        if output_format == "sqlite":
            conn = storage
            for result in results:
                conn.execute("""
                    INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.submission_id,
                    result.date_folder,
                    result.predicted_label,
                    result.predicted_class,
                    result.probability_relevant,
                    result.confidence,
                    result.processing_time,
                    result.timestamp.isoformat(),
                    result.error
                ))
            conn.commit()
        
        else:
            # For other formats, accumulate results
            storage.extend(results)
    
    def _finalize_storage(self, storage: Any, output_format: str, output_path: Path):
        """Finalize storage and save results"""
        if output_format == "sqlite":
            storage.close()
        
        elif output_format == "parquet":
            df = pd.DataFrame([asdict(result) for result in storage])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.to_parquet(output_path / "predictions.parquet", index=False)
        
        elif output_format == "csv":
            df = pd.DataFrame([asdict(result) for result in storage])
            df.to_csv(output_path / "predictions.csv", index=False)
        
        elif output_format == "json":
            results_dict = [asdict(result) for result in storage]
            # Convert datetime to string
            for result in results_dict:
                result['timestamp'] = result['timestamp'].isoformat()
            
            with open(output_path / "predictions.json", 'w') as f:
                json.dump(results_dict, f, indent=2)
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the model being used"""
        return {
            'featurizer_type': type(self.predictor.featurizer).__name__,
            'model_type': type(self.predictor.model).__name__,
            'feature_count': len(self.predictor.feature_names) if self.predictor.feature_names else 0,
            'uses_scaler': self.predictor.scaler is not None,
            'min_confidence_threshold': self.min_confidence_threshold
        }
    
    def _save_summary(self, summary: BatchPredictionResults, output_path: Path):
        """Save processing summary"""
        summary_dict = asdict(summary)
        summary_dict['timestamp'] = summary_dict['timestamp'].isoformat()
        
        with open(output_path / "processing_summary.json", 'w') as f:
            json.dump(summary_dict, f, indent=2)

# =============================================================================
# Analysis and Filtering Classes
# =============================================================================

class PredictionAnalyzer:
    """Analyze and filter prediction results"""
    
    def __init__(self, results_path: Path, format_type: str = "parquet"):
        """
        Initialize analyzer
        
        Args:
            results_path: Path to prediction results
            format_type: Format of results ("parquet", "csv", "sqlite")
        """
        self.results_path = Path(results_path)
        self.format_type = format_type
        self.df = self._load_results()
    
    def _load_results(self) -> pd.DataFrame:
        """Load results into DataFrame"""
        if self.format_type == "parquet":
            return pd.read_parquet(self.results_path / "predictions.parquet")
        elif self.format_type == "csv":
            return pd.read_csv(self.results_path / "predictions.csv")
        elif self.format_type == "sqlite":
            conn = sqlite3.connect(str(self.results_path / "predictions.db"))
            return pd.read_sql_query("SELECT * FROM predictions", conn)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")
    
    def get_relevant_submissions(self, min_confidence: float = 0.7) -> pd.DataFrame:
        """Get submissions predicted as relevant with high confidence"""
        return self.df[
            (self.df['predicted_label'] == 1) & 
            (self.df['confidence'] >= min_confidence) &
            (self.df['error'].isna())
        ].copy()
    
    def get_daily_stats(self) -> pd.DataFrame:
        """Get daily statistics"""
        daily_stats = self.df.groupby('date_folder').agg({
            'submission_id': 'count',
            'predicted_label': ['sum', 'mean'],
            'confidence': 'mean',
            'error': lambda x: x.notna().sum()
        }).round(3)
        
        daily_stats.columns = ['total_submissions', 'relevant_count', 'relevance_rate', 'avg_confidence', 'error_count']
        return daily_stats.reset_index()
    
    def export_relevant_for_sentiment(self, output_path: Path, min_confidence: float = 0.8):
        """Export relevant submissions for sentiment analysis"""
        relevant_df = self.get_relevant_submissions(min_confidence)
        
        # Create export format suitable for sentiment pipeline
        export_data = relevant_df[['submission_id', 'date_folder', 'confidence']].copy()
        export_data['file_path'] = export_data.apply(
            lambda row: f"{row['date_folder']}/{row['submission_id']}.json", axis=1
        )
        
        export_data.to_csv(output_path, index=False)
        print(f"Exported {len(export_data)} relevant submissions to {output_path}")