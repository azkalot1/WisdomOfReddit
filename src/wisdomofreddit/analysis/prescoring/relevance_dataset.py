import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from .classes import LabeledSubmission, LabeledFeaturizedSample, TrainingDatasetMetadata
from .submission_featurizer import SubmissionFeaturizer
from tqdm.auto import tqdm

class TrainingDataset:
    """
    Complete training dataset with features, labels, and featurizer
    
    This class stores everything needed for training:
    - Labeled submissions
    - Extracted features
    - The featurizer used
    - Metadata about the dataset
    """
    
    def __init__(self, 
                 dataset_name: str,
                 featurizer: 'SubmissionFeaturizer',
                 data_sources: Optional[List[str]] = None):
        """
        Initialize training dataset
        
        Args:
            dataset_name: Name for this dataset
            featurizer: The featurizer to use for feature extraction
            data_sources: List of data sources (e.g., ['reddit_dump_2020', 'manual_collection'])
        """
        self.dataset_name = dataset_name
        self.featurizer = featurizer
        self.data_sources = data_sources or []
        
        # Storage for labeled data
        self.labeled_submissions: List[LabeledSubmission] = []
        self.featurized_samples: List[LabeledFeaturizedSample] = []
        
        # Computed properties
        self._features_extracted = False
        self._feature_matrix: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._metadata: Optional[TrainingDatasetMetadata] = None
    
    def add_labeled_submission(self, 
                              submission_data: Dict[str, Any],
                              label: int,
                              labeler: Optional[str] = 'default',
                              confidence: float = 5.0,
                              notes: str = "") -> None:
        """
        Add a labeled submission to the dataset
        
        Args:
            submission_data: Original submission JSON
            label: 0 (irrelevant) or 1 (relevant)
            labeler: Who labeled this submission
            confidence: Confidence in the label (1-5)
            notes: Optional notes about the labeling
        """
        if label not in [0, 1]:
            raise ValueError("Label must be 0 or 1")
        
        labeled_sub = LabeledSubmission(
            submission_id=submission_data.get('submission_id', f'unknown_{len(self.labeled_submissions)}'),
            subreddit=submission_data.get('subreddit', 'unknown'),
            date=submission_data.get('date', 'unknown'),
            submission_data=submission_data,
            label=label,
            labeler=labeler,
            confidence=confidence,
            timestamp=datetime.now(),
            notes=notes
        )
        
        self.labeled_submissions.append(labeled_sub)
        self._features_extracted = False  # Need to re-extract features
    
    def add_labeled_submissions_batch(self, 
                                    labeled_data: List[Dict[str, Any]]) -> None:
        """
        Add multiple labeled submissions at once
        
        Args:
            labeled_data: List of dictionaries with keys:
                - submission_data (required): Dict with submission JSON
                - label (required): int, 0 or 1
                - labeler (optional): str, defaults to "default"
                - confidence (optional): float, defaults to 5.0
                - notes (optional): str, defaults to ""
        
        Examples:
            dataset.add_labeled_submissions_batch([
                {
                    'submission_data': submission1,
                    'label': 1
                },
                {
                    'submission_data': submission2,
                    'label': 0,
                    'labeler': 'expert',
                    'confidence': 4.5,
                    'notes': 'Clear psychology post'
                }
            ])
        """
        for item in labeled_data:
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict, got {type(item)}")
            
            if 'submission_data' not in item:
                raise ValueError("Missing required key 'submission_data'")
            if 'label' not in item:
                raise ValueError("Missing required key 'label'")
            
            # Unpack with defaults
            self.add_labeled_submission(
                submission_data=item['submission_data'],
                label=item['label'],
                labeler=item.get('labeler', 'default'),
                confidence=item.get('confidence', 5.0),
                notes=item.get('notes', '')
            )

    def add_from_tuples(self, 
                    tuple_data: List[Tuple[Dict[str, Any], int, str]]) -> None:
        """
        Convenience method to convert from tuple format
        
        Args:
            tuple_data: List of (submission_data, label, labeler) tuples
        """
        labeled_data = [
            {
                'submission_data': submission_data,
                'label': label,
                'labeler': labeler
            }
            for submission_data, label, labeler in tuple_data
        ]
        self.add_labeled_submissions_batch(labeled_data)

    def add_simple_labels(self, 
                        submissions_and_labels: List[Tuple[Dict[str, Any], int]],
                        labeler: str = "default") -> None:
        """
        Convenience method for simple (submission, label) pairs
        
        Args:
            submissions_and_labels: List of (submission_data, label) tuples
            labeler: Labeler name for all submissions
        """
        labeled_data = [
            {
                'submission_data': submission_data,
                'label': label,
                'labeler': labeler
            }
            for submission_data, label in submissions_and_labels
        ]
        self.add_labeled_submissions_batch(labeled_data)

    def add_from_labeling_interface(self, labeling_file: Path) -> None:
        """
        Load directly from labeling interface JSON output
        
        Args:
            labeling_file: Path to JSON file from labeling interface
        """
        with open(labeling_file, 'r') as f:
            labeled_json = json.load(f)
        
        labeled_data = [
            {
                'submission_data': item['submission_data'],
                'label': 1 if item['label'] == 'relevant' else 0,
                'labeler': item.get('labeler', 'default'),
                'confidence': item.get('confidence', 5.0),
                'notes': item.get('notes', '')
            }
            for item in labeled_json
        ]
        
        self.add_labeled_submissions_batch(labeled_data)
    
    def extract_features(self, force_recompute: bool = False) -> None:
        """
        Extract features from all labeled submissions
        
        Args:
            force_recompute: Whether to recompute features even if already extracted
        """
        if self._features_extracted and not force_recompute:
            print("Features already extracted. Use force_recompute=True to recompute.")
            return
        
        print(f"Extracting features from {len(self.labeled_submissions)} labeled submissions...")
        
        self.featurized_samples = []
        
        for labeled_sub in tqdm(self.labeled_submissions):
            try:
                # Extract features using the featurizer
                feature_result = self.featurizer.extract_features(labeled_sub.submission_data)
                
                # Create featurized sample
                featurized_sample = LabeledFeaturizedSample(
                    submission_id=labeled_sub.submission_id,
                    subreddit=labeled_sub.subreddit,
                    date=labeled_sub.date,
                    features=feature_result.features,
                    feature_names=feature_result.feature_names,
                    feature_vector=feature_result.feature_vector,
                    label=labeled_sub.label,
                    metadata={
                        'labeler': labeled_sub.labeler,
                        'confidence': labeled_sub.confidence,
                        'timestamp': labeled_sub.timestamp.isoformat(),
                        'notes': labeled_sub.notes,
                        'extraction_metadata': feature_result.metadata
                    }
                )
                
                self.featurized_samples.append(featurized_sample)
                
            except Exception as e:
                print(f"Failed to extract features for {labeled_sub.submission_id}: {e}")
                # Could add error handling here (skip, use zeros, etc.)
        
        self._features_extracted = True
        self._invalidate_computed_properties()
        
        print(f"✓ Extracted features for {len(self.featurized_samples)} submissions")
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get training data in format ready for sklearn
        
        Returns:
            Tuple of (X, y, submission_ids)
            - X: Feature matrix (n_samples, n_features)
            - y: Labels (n_samples,)
            - submission_ids: List of submission IDs
        """
        if not self._features_extracted:
            self.extract_features()
        
        if self._feature_matrix is None or self._labels is None:
            # Build feature matrix and labels
            feature_vectors = [sample.feature_vector for sample in self.featurized_samples]
            labels = [sample.label for sample in self.featurized_samples]
            
            self._feature_matrix = np.array(feature_vectors)
            self._labels = np.array(labels)
        
        submission_ids = [sample.submission_id for sample in self.featurized_samples]
        
        return self._feature_matrix, self._labels, submission_ids
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from the featurizer"""
        return self.featurizer.get_feature_names()
    
    def get_metadata(self) -> TrainingDatasetMetadata:
        """Get dataset metadata"""
        if self._metadata is None:
            self._compute_metadata()
        return self._metadata
    
    def save(self, folder_path: Path, overwrite: bool = False) -> None:
        """
        Save the complete training dataset to a folder
        
        Args:
            folder_path: Path to folder where dataset will be saved
                        (e.g., "data/training_datasets/dataset_v1")
            overwrite: If True, overwrite existing files. If False, raise error if folder exists
        
        Creates folder structure:
            folder_path/
            ├── dataset.pkl          # Complete dataset with featurizer
            ├── metadata.json        # Human-readable metadata
            ├── features.npz         # Feature matrix (X, y) for quick loading
            ├── labeled_data.json    # Raw labeled submissions (backup)
            ├── feature_names.txt    # Feature names list
            └── summary.txt          # Dataset summary
        
        Raises:
            FileExistsError: If folder exists and overwrite=False
        """
        folder_path = Path(folder_path)
        
        # Check if folder exists and handle overwrite logic
        if folder_path.exists():
            if not overwrite:
                # Check if it contains dataset files
                dataset_files = ['dataset.pkl', 'metadata.json', 'features.npz', 'labeled_data.json']
                existing_files = [f for f in dataset_files if (folder_path / f).exists()]
                
                if existing_files:
                    raise FileExistsError(
                        f"Dataset folder already exists: {folder_path}\n"
                        f"Existing files: {existing_files}\n"
                        f"Use overwrite=True to overwrite, or choose a different folder."
                    )
            else:
                print(f"⚠ Overwriting existing dataset in: {folder_path}")
        
        # Create folder (this is safe even if it exists)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving training dataset to: {folder_path}")
        
        # 1. Save the complete dataset as pickle (includes featurizer)
        dataset_path = folder_path / "dataset.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  ✓ dataset.pkl (complete dataset with featurizer)")
        
        # 2. Save human-readable metadata as JSON
        metadata_path = folder_path / "metadata.json"
        metadata = self.get_metadata()
        metadata_dict = asdict(metadata)
        
        # Convert datetime to string for JSON serialization
        metadata_dict['creation_date'] = metadata_dict['creation_date'].isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        print(f"  ✓ metadata.json (dataset metadata)")
        
        # 3. Save feature matrix as numpy array (for quick loading)
        if self._features_extracted:
            X, y, submission_ids = self.get_training_data()
            features_path = folder_path / "features.npz"
            np.savez(features_path, X=X, y=y, submission_ids=submission_ids)
            print(f"  ✓ features.npz (feature matrix: {X.shape})")
        else:
            print(f"  ⚠ features.npz (skipped - features not extracted)")
        
        # 4. Save raw labeled data as backup (for inspection/debugging)
        labeled_data_path = folder_path / "labeled_data.json"
        labeled_data_backup = []
        for labeled_sub in self.labeled_submissions:
            labeled_data_backup.append({
                'submission_id': labeled_sub.submission_id,
                'submission_data': labeled_sub.submission_data,
                'label': labeled_sub.label,
                'labeler': labeled_sub.labeler,
                'confidence': labeled_sub.confidence,
                'timestamp': labeled_sub.timestamp.isoformat(),
                'notes': labeled_sub.notes
            })
        
        with open(labeled_data_path, 'w') as f:
            json.dump(labeled_data_backup, f, indent=2)
        print(f"  ✓ labeled_data.json ({len(labeled_data_backup)} labeled submissions)")
        
        # 5. Save feature names as text file (for easy inspection)
        feature_names_path = folder_path / "feature_names.txt"
        feature_names = self.get_feature_names()
        with open(feature_names_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"  ✓ feature_names.txt ({len(feature_names)} features)")
        
        # 6. Save dataset summary
        summary_path = folder_path / "summary.txt"
        with open(summary_path, 'w') as f:
            summary = self.get_summary()
            f.write(f"Training Dataset Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Dataset Name: {summary['dataset_name']}\n")
            f.write(f"Total Samples: {summary['total_samples']}\n")
            f.write(f"Positive Samples: {summary['positive_samples']} ({summary['class_balance']:.1%})\n")
            f.write(f"Negative Samples: {summary['negative_samples']} ({1-summary['class_balance']:.1%})\n")
            f.write(f"Feature Count: {summary['feature_count']}\n")
            f.write(f"Labelers: {', '.join(summary['labelers'])}\n")
            f.write(f"Average Confidence: {summary['avg_confidence']:.1f}/5.0\n")
            f.write(f"Data Sources: {', '.join(summary['data_sources'])}\n")
            f.write(f"Features Extracted: {summary['features_extracted']}\n")
            f.write(f"Saved: {datetime.now().isoformat()}\n")
        print(f"  ✓ summary.txt (human-readable summary)")
        
        print(f"✅ Dataset saved successfully!")

    def save_backup(self, folder_path: Path, backup_suffix: str = None) -> Path:
        """
        Save dataset with automatic backup naming if folder exists
        
        Args:
            folder_path: Desired folder path
            backup_suffix: Optional suffix for backup (default: timestamp)
            
        Returns:
            Path where dataset was actually saved
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            # Folder doesn't exist, save normally
            self.save(folder_path, overwrite=False)
            return folder_path
        
        # Folder exists, create backup name
        if backup_suffix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_suffix = timestamp
        
        backup_path = folder_path.parent / f"{folder_path.name}_{backup_suffix}"
        
        print(f"📁 Folder exists, saving to backup location: {backup_path}")
        self.save(backup_path, overwrite=False)
        return backup_path

    def save_version(self, base_folder_path: Path, version: str = None) -> Path:
        """
        Save dataset with version numbering
        
        Args:
            base_folder_path: Base path (e.g., "data/datasets/reddit_relevance")
            version: Version string (e.g., "v1", "v2.1"). If None, auto-increment
            
        Returns:
            Path where dataset was saved
        """
        base_folder_path = Path(base_folder_path)
        
        if version is None:
            # Auto-increment version
            version_num = 1
            while True:
                version_path = base_folder_path.parent / f"{base_folder_path.name}_v{version_num}"
                if not version_path.exists():
                    break
                version_num += 1
            version = f"v{version_num}"
        
        versioned_path = base_folder_path.parent / f"{base_folder_path.name}_{version}"
        
        print(f"📦 Saving dataset version: {version}")
        self.save(versioned_path, overwrite=False)
        return versioned_path
    
    @classmethod
    def load(cls, folder_path: Path) -> 'TrainingDataset':
        """
        Load training dataset from folder
        
        Args:
            folder_path: Path to folder containing dataset files
            
        Returns:
            Loaded TrainingDataset
            
        Expected folder structure:
            folder_path/
            ├── dataset.pkl          # Main dataset file
            ├── metadata.json        # Metadata (optional, for verification)
            ├── features.npz         # Feature matrix (optional, for quick access)
            ├── labeled_data.json    # Raw data (optional, for backup)
            └── feature_names.txt    # Feature names (optional, for inspection)
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
        
        # Load the main dataset file
        dataset_path = folder_path / "dataset.pkl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Main dataset file not found: {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Verify metadata if available
        metadata_path = folder_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
            
            print(f"✓ Loaded dataset: {saved_metadata['dataset_name']}")
            print(f"  Created: {saved_metadata['creation_date']}")
            print(f"  Samples: {saved_metadata['total_samples']} "
                f"({saved_metadata['positive_samples']} positive, "
                f"{saved_metadata['negative_samples']} negative)")
            print(f"  Features: {saved_metadata['feature_count']}")
            print(f"  Labelers: {', '.join(saved_metadata['labelers'])}")
        else:
            print(f"✓ Loaded dataset from: {folder_path}")
            print(f"  - {len(dataset.labeled_submissions)} labeled submissions")
            print(f"  - {len(dataset.featurized_samples)} featurized samples")
            print(f"  - Features extracted: {dataset._features_extracted}")
        
        return dataset

    @classmethod
    def load_quick(cls, folder_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Quick load of just the feature matrix and labels (without full dataset)
        
        Args:
            folder_path: Path to folder containing dataset files
            
        Returns:
            Tuple of (X, y, submission_ids, feature_names)
            
        This is useful when you just need the training data without the full dataset object
        """
        folder_path = Path(folder_path)
        
        # Load feature matrix
        features_path = folder_path / "features.npz"
        if not features_path.exists():
            raise FileNotFoundError(f"Feature matrix not found: {features_path}")
        
        data = np.load(features_path)
        X = data['X']
        y = data['y']
        submission_ids = data['submission_ids'].tolist()
        
        # Load feature names
        feature_names_path = folder_path / "feature_names.txt"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        print(f"✓ Quick loaded training data: X={X.shape}, y={y.shape}")
        
        return X, y, submission_ids, feature_names

    def get_folder_info(folder_path: Path) -> Dict[str, Any]:
        """
        Get information about a dataset folder without loading it
        
        Args:
            folder_path: Path to dataset folder
            
        Returns:
            Dictionary with folder information
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return {'exists': False, 'error': 'Folder not found'}
        
        info = {'exists': True, 'files': {}}
        
        # Check for expected files
        expected_files = {
            'dataset.pkl': 'Main dataset file',
            'metadata.json': 'Dataset metadata',
            'features.npz': 'Feature matrix',
            'labeled_data.json': 'Raw labeled data',
            'feature_names.txt': 'Feature names',
            'summary.txt': 'Dataset summary'
        }
        
        for filename, description in expected_files.items():
            filepath = folder_path / filename
            info['files'][filename] = {
                'exists': filepath.exists(),
                'description': description,
                'size_mb': filepath.stat().st_size / (1024*1024) if filepath.exists() else 0
            }
        
        # Load metadata if available
        metadata_path = folder_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                info['metadata'] = metadata
            except:
                info['metadata'] = {'error': 'Failed to load metadata'}
        
        return info
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the dataset"""
        if not self._features_extracted:
            self.extract_features()
        
        X, y, _ = self.get_training_data()
        
        return {
            'dataset_name': self.dataset_name,
            'total_samples': len(y),
            'positive_samples': int(np.sum(y)),
            'negative_samples': int(len(y) - np.sum(y)),
            'class_balance': float(np.mean(y)),
            'feature_count': X.shape[1],
            'labelers': list(set(sub.labeler for sub in self.labeled_submissions)),
            'avg_confidence': np.mean([sub.confidence for sub in self.labeled_submissions]),
            'data_sources': self.data_sources,
            'features_extracted': self._features_extracted
        }
    
    def print_summary(self) -> None:
        """Print a nice summary of the dataset"""
        summary = self.get_summary()
        
        print(f"\n{'='*50}")
        print(f"Training Dataset: {summary['dataset_name']}")
        print(f"{'='*50}")
        print(f"Total samples: {summary['total_samples']}")
        print(f"  - Relevant (1): {summary['positive_samples']} ({summary['class_balance']:.1%})")
        print(f"  - Irrelevant (0): {summary['negative_samples']} ({1-summary['class_balance']:.1%})")
        print(f"Features: {summary['feature_count']}")
        print(f"Labelers: {', '.join(summary['labelers'])}")
        print(f"Avg confidence: {summary['avg_confidence']:.1f}/5.0")
        print(f"Data sources: {', '.join(summary['data_sources'])}")
        print(f"Features extracted: {summary['features_extracted']}")
    
    def _compute_metadata(self) -> None:
        """Compute dataset metadata"""
        if not self._features_extracted:
            self.extract_features()
        
        X, y, _ = self.get_training_data()
        
        # Compute featurizer hash for reproducibility
        featurizer_str = str(self.featurizer.get_feature_names())
        featurizer_hash = hashlib.md5(featurizer_str.encode()).hexdigest()[:8]
        
        self._metadata = TrainingDatasetMetadata(
            dataset_name=self.dataset_name,
            creation_date=datetime.now(),
            total_samples=len(y),
            positive_samples=int(np.sum(y)),
            negative_samples=int(len(y) - np.sum(y)),
            labelers=list(set(sub.labeler for sub in self.labeled_submissions)),
            featurizer_hash=featurizer_hash,
            feature_count=X.shape[1],
            feature_names=self.get_feature_names(),
            data_sources=self.data_sources,
            notes=""
        )
    
    def _invalidate_computed_properties(self) -> None:
        """Invalidate cached computed properties"""
        self._feature_matrix = None
        self._labels = None
        self._metadata = None

# =============================================================================
# Convenience Functions
# =============================================================================

def create_training_dataset_from_labels(
    dataset_name: str,
    labeled_submissions_file: Path,
    featurizer: 'SubmissionFeaturizer',
    data_sources: Optional[List[str]] = None
) -> TrainingDataset:
    """
    Create training dataset from a file of labeled submissions
    
    Args:
        dataset_name: Name for the dataset
        labeled_submissions_file: JSON file with labeled submissions
        featurizer: Featurizer to use
        data_sources: List of data sources
        
    Returns:
        TrainingDataset ready for training
    """
    # Load labeled submissions from file
    with open(labeled_submissions_file, 'r') as f:
        labeled_data = json.load(f)
    
    # Create dataset
    dataset = TrainingDataset(dataset_name, featurizer, data_sources)
    
    # Add labeled submissions
    for item in labeled_data:
        dataset.add_labeled_submission(
            submission_data=item['submission_data'],
            label=item['label'],
            labeler=item.get('labeler', 'unknown'),
            confidence=item.get('confidence', 5.0),
            notes=item.get('notes', '')
        )
    
    # Extract features
    dataset.extract_features()
    
    return dataset

def merge_training_datasets(datasets: List[TrainingDataset], 
                          new_name: str) -> TrainingDataset:
    """
    Merge multiple training datasets into one
    
    Args:
        datasets: List of TrainingDataset objects to merge
        new_name: Name for the merged dataset
        
    Returns:
        Merged TrainingDataset
    """
    if not datasets:
        raise ValueError("No datasets provided")
    
    # Use featurizer from first dataset (assume they're compatible)
    merged_dataset = TrainingDataset(
        dataset_name=new_name,
        featurizer=datasets[0].featurizer,
        data_sources=[]
    )
    
    # Merge all labeled submissions
    for dataset in datasets:
        merged_dataset.labeled_submissions.extend(dataset.labeled_submissions)
        merged_dataset.data_sources.extend(dataset.data_sources)
    
    # Remove duplicates from data sources
    merged_dataset.data_sources = list(set(merged_dataset.data_sources))
    
    # Extract features
    merged_dataset.extract_features()
    
    return merged_dataset
