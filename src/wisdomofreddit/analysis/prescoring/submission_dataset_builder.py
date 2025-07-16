import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import glob
from datetime import datetime
from .relevance_dataset import TrainingDataset

class SubmissionLookup:
    """
    Helper class to lookup submissions by ID from organized folder structure
    """
    
    def __init__(self, base_data_path: Path):
        """
        Initialize with base path to reddit data
        
        Args:
            base_data_path: Path to folder containing date-organized submissions
                           e.g., "reddit_comments" containing "20200103", "20200104", etc.
        """
        self.base_data_path = Path(base_data_path)
        self._submission_cache = {}  # Cache for faster lookups
        self._build_index()
    
    def _build_index(self) -> None:
        """Build index of submission_id -> file_path for faster lookups"""
        print(f"🔍 Building submission index from {self.base_data_path}...")
        
        # Find all JSON files in date folders
        pattern = str(self.base_data_path / "*" / "*.json")
        json_files = glob.glob(pattern)
        
        print(f"   Found {len(json_files)} JSON files")
        
        # Build index (submission_id -> file_path)
        self._submission_index = {}
        
        for i, file_path in enumerate(json_files):
            try:
                # Extract submission_id from filename (assuming filename is submission_id.json)
                submission_id = Path(file_path).stem
                self._submission_index[submission_id] = file_path
                
                if (i + 1) % 1000 == 0:
                    print(f"   Indexed {i + 1}/{len(json_files)} files...")
                    
            except Exception as e:
                print(f"   Warning: Could not index {file_path}: {e}")
        
        print(f"✅ Index built: {len(self._submission_index)} submissions indexed")
    
    def find_submission(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """
        Find and load submission by ID
        
        Args:
            submission_id: The submission ID to find
            
        Returns:
            Submission data dict or None if not found
        """
        # Check cache first
        if submission_id in self._submission_cache:
            return self._submission_cache[submission_id]
        
        # Check index
        if submission_id not in self._submission_index:
            print(f"❌ Submission {submission_id} not found in index")
            return None
        
        # Load from file
        file_path = self._submission_index[submission_id]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Cache for future use
            self._submission_cache[submission_id] = data
            return data
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def find_multiple_submissions(self, submission_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Find multiple submissions at once
        
        Args:
            submission_ids: List of submission IDs to find
            
        Returns:
            Dict mapping submission_id -> submission_data (or None if not found)
        """
        results = {}
        
        print(f"🔍 Looking up {len(submission_ids)} submissions...")
        
        for i, submission_id in enumerate(submission_ids):
            results[submission_id] = self.find_submission(submission_id)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(submission_ids)} lookups")
        
        found_count = sum(1 for v in results.values() if v is not None)
        print(f"✅ Found {found_count}/{len(submission_ids)} submissions")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed data"""
        return {
            'total_indexed': len(self._submission_index),
            'cached_submissions': len(self._submission_cache),
            'base_path': str(self.base_data_path),
            'sample_submission_ids': list(self._submission_index.keys())[:10]
        }

class DatasetBuilder:
    """
    Helper class to build training datasets with submission lookup and duplicate prevention
    """
    
    def __init__(self, 
                 dataset: 'TrainingDataset', 
                 submission_lookup: SubmissionLookup):
        """
        Initialize dataset builder
        
        Args:
            dataset: TrainingDataset to add submissions to
            submission_lookup: SubmissionLookup for finding submissions
        """
        self.dataset = dataset
        self.lookup = submission_lookup
        self._added_submissions = set()  # Track added submission IDs
        self._build_existing_index()
    
    def _build_existing_index(self) -> None:
        """Build index of already added submissions"""
        print("🔍 Building index of existing submissions in dataset...")
        
        for labeled_sub in self.dataset.labeled_submissions:
            self._added_submissions.add(labeled_sub.submission_id)
        
        if self._added_submissions:
            print(f"   Found {len(self._added_submissions)} existing submissions")
        else:
            print("   No existing submissions found")
    
    def add(self, 
            submission_id: str, 
            label: int, 
            labeler: str = "manual",
            confidence: float = 5.0,
            notes: str = "",
            allow_duplicate: bool = False) -> Dict[str, Any]:
        """
        Add a single submission to dataset by ID
        
        Args:
            submission_id: ID of submission to add
            label: 0 (irrelevant) or 1 (relevant)
            labeler: Who labeled this submission
            confidence: Confidence in the label (1-5)
            notes: Optional notes
            allow_duplicate: If True, allow adding duplicate submissions
            
        Returns:
            Dict with result information
        """
        # Check for duplicate
        if not allow_duplicate and submission_id in self._added_submissions:
            print(f"⚠️  Submission {submission_id} already exists in dataset - skipping")
            return {
                'success': False,
                'reason': 'duplicate',
                'submission_id': submission_id,
                'message': 'Submission already exists in dataset'
            }
        
        # Find submission
        submission_data = self.lookup.find_submission(submission_id)
        
        if submission_data is None:
            print(f"❌ Cannot add {submission_id}: submission not found")
            return {
                'success': False,
                'reason': 'not_found',
                'submission_id': submission_id,
                'message': 'Submission not found in data files'
            }
        
        # Add to dataset
        labeled_data = [{
            'submission_data': submission_data,
            'label': label,
            'labeler': labeler,
            'confidence': confidence,
            'notes': notes
        }]
        
        self.dataset.add_labeled_submissions_batch(labeled_data)
        
        # Track as added
        self._added_submissions.add(submission_id)
        
        print(f"✅ Added {submission_id} with label {label}")
        return {
            'success': True,
            'reason': 'added',
            'submission_id': submission_id,
            'label': label,
            'message': f'Successfully added with label {label}'
        }
    
    def add_batch(self, 
                  labeled_submissions: List[Tuple[str, int]], 
                  labeler: str = "batch_processor",
                  confidence: float = 5.0,
                  allow_duplicates: bool = False) -> Dict[str, Any]:
        """
        Add multiple submissions by ID with duplicate checking
        
        Args:
            labeled_submissions: List of (submission_id, label) tuples
            labeler: Who labeled these submissions
            confidence: Confidence for all labels
            allow_duplicates: If True, allow adding duplicate submissions
            
        Returns:
            Dict with detailed success/failure statistics
        """
        print(f"📦 Adding batch of {len(labeled_submissions)} submissions...")
        
        # Separate duplicates and new submissions
        new_submissions = []
        duplicates = []
        
        for submission_id, label in labeled_submissions:
            if not allow_duplicates and submission_id in self._added_submissions:
                duplicates.append((submission_id, label))
            else:
                new_submissions.append((submission_id, label))
        
        if duplicates:
            print(f"⚠️  Skipping {len(duplicates)} duplicate submissions")
            if len(duplicates) <= 5:
                duplicate_ids = [sub_id for sub_id, _ in duplicates]
                print(f"   Duplicates: {duplicate_ids}")
            else:
                print(f"   First 5 duplicates: {[sub_id for sub_id, _ in duplicates[:5]]}")
        
        if not new_submissions:
            print("ℹ️  No new submissions to add")
            return {
                'total_requested': len(labeled_submissions),
                'successfully_added': 0,
                'duplicates_skipped': len(duplicates),
                'failed_lookups': 0,
                'failed_submission_ids': [],
                'duplicate_submission_ids': [sub_id for sub_id, _ in duplicates],
                'success_rate': 0.0
            }
        
        # Lookup new submissions
        submission_ids = [sub_id for sub_id, _ in new_submissions]
        found_submissions = self.lookup.find_multiple_submissions(submission_ids)
        
        # Prepare data for successful lookups
        successful_additions = []
        failed_lookups = []
        
        for submission_id, label in new_submissions:
            submission_data = found_submissions.get(submission_id)
            
            if submission_data is not None:
                successful_additions.append({
                    'submission_data': submission_data,
                    'label': label,
                    'labeler': labeler,
                    'confidence': confidence,
                    'notes': f'Batch added from ID lookup'
                })
                # Track as added
                self._added_submissions.add(submission_id)
            else:
                failed_lookups.append(submission_id)
        
        # Add successful ones to dataset
        if successful_additions:
            self.dataset.add_labeled_submissions_batch(successful_additions)
        
        # Report results
        results = {
            'total_requested': len(labeled_submissions),
            'successfully_added': len(successful_additions),
            'duplicates_skipped': len(duplicates),
            'failed_lookups': len(failed_lookups),
            'failed_submission_ids': failed_lookups,
            'duplicate_submission_ids': [sub_id for sub_id, _ in duplicates],
            'success_rate': len(successful_additions) / len(labeled_submissions) if labeled_submissions else 0
        }
        
        print(f"✅ Batch complete: {results['successfully_added']}/{results['total_requested']} added")
        if results['duplicates_skipped'] > 0:
            print(f"⚠️  Skipped {results['duplicates_skipped']} duplicates")
        if failed_lookups:
            print(f"❌ Failed to find: {failed_lookups[:5]}{'...' if len(failed_lookups) > 5 else ''}")
        
        return results
    
    def add_from_file(self, 
                      labels_file: Path, 
                      labeler: str = "file_import",
                      allow_duplicates: bool = False) -> Dict[str, Any]:
        """
        Add submissions from a file containing submission_id,label pairs
        
        Args:
            labels_file: Path to CSV/JSON file with submission IDs and labels
            labeler: Who labeled these submissions
            allow_duplicates: If True, allow adding duplicate submissions
            
        Returns:
            Dict with import statistics
        """
        labels_file = Path(labels_file)
        
        if labels_file.suffix.lower() == '.csv':
            # Load from CSV
            import pandas as pd
            df = pd.read_csv(labels_file)
            
            # Expect columns: submission_id, label
            if 'submission_id' not in df.columns or 'label' not in df.columns:
                raise ValueError("CSV must have 'submission_id' and 'label' columns")
            
            labeled_submissions = list(zip(df['submission_id'], df['label']))
            
        elif labels_file.suffix.lower() == '.json':
            # Load from JSON
            with open(labels_file, 'r') as f:
                data = json.load(f)
            
            # Expect list of {"submission_id": "...", "label": ...} objects
            labeled_submissions = [(item['submission_id'], item['label']) for item in data]
            
        else:
            raise ValueError("File must be .csv or .json")
        
        print(f"📁 Loading {len(labeled_submissions)} labels from {labels_file}")
        return self.add_batch(labeled_submissions, labeler, allow_duplicates=allow_duplicates)
    
    def check_duplicates(self, submission_ids: List[str]) -> Dict[str, List[str]]:
        """
        Check which submission IDs are already in the dataset
        
        Args:
            submission_ids: List of submission IDs to check
            
        Returns:
            Dict with 'existing' and 'new' lists
        """
        existing = []
        new = []
        
        for sub_id in submission_ids:
            if sub_id in self._added_submissions:
                existing.append(sub_id)
            else:
                new.append(sub_id)
        
        return {
            'existing': existing,
            'new': new,
            'total_checked': len(submission_ids),
            'existing_count': len(existing),
            'new_count': len(new)
        }
    
    def get_added_submissions(self) -> List[str]:
        """Get list of all submission IDs that have been added"""
        return list(self._added_submissions)
    
    def remove_submission(self, submission_id: str) -> bool:
        """
        Remove a submission from the dataset (if it exists)
        
        Args:
            submission_id: ID of submission to remove
            
        Returns:
            True if removed, False if not found
        """
        # Find and remove from dataset
        original_count = len(self.dataset.labeled_submissions)
        
        self.dataset.labeled_submissions = [
            sub for sub in self.dataset.labeled_submissions 
            if sub.submission_id != submission_id
        ]
        
        new_count = len(self.dataset.labeled_submissions)
        removed = original_count > new_count
        
        if removed:
            # Remove from tracking set
            self._added_submissions.discard(submission_id)
            # Invalidate cached features
            self.dataset._features_extracted = False
            print(f"✅ Removed submission {submission_id}")
        else:
            print(f"⚠️  Submission {submission_id} not found in dataset")
        
        return removed
    
    def update_submission_label(self, 
                               submission_id: str, 
                               new_label: int,
                               labeler: str = "updated",
                               confidence: float = 5.0,
                               notes: str = "Label updated") -> bool:
        """
        Update the label of an existing submission
        
        Args:
            submission_id: ID of submission to update
            new_label: New label (0 or 1)
            labeler: Who updated the label
            confidence: New confidence
            notes: Notes about the update
            
        Returns:
            True if updated, False if not found
        """
        # Find submission in dataset
        for labeled_sub in self.dataset.labeled_submissions:
            if labeled_sub.submission_id == submission_id:
                # Update the label
                old_label = labeled_sub.label
                labeled_sub.label = new_label
                labeled_sub.labeler = labeler
                labeled_sub.confidence = confidence
                labeled_sub.notes = notes
                labeled_sub.timestamp = datetime.now()
                
                # Invalidate cached features
                self.dataset._features_extracted = False
                
                print(f"✅ Updated {submission_id}: label {old_label} → {new_label}")
                return True
        
        print(f"⚠️  Submission {submission_id} not found in dataset")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the builder and dataset"""
        return {
            'total_submissions_tracked': len(self._added_submissions),
            'dataset_submission_count': len(self.dataset.labeled_submissions),
            'lookup_stats': self.lookup.get_stats(),
            'sample_added_ids': list(self._added_submissions)[:10]
        }

# =============================================================================
# Convenience Functions
# =============================================================================

def create_dataset_builder(dataset: 'TrainingDataset', 
                          base_data_path: Path) -> DatasetBuilder:
    """
    Create a DatasetBuilder with submission lookup
    
    Args:
        dataset: TrainingDataset to add to
        base_data_path: Path to reddit data folder
        
    Returns:
        DatasetBuilder ready to use
    """
    lookup = SubmissionLookup(base_data_path)
    return DatasetBuilder(dataset, lookup)

def quick_add_submissions(dataset: 'TrainingDataset',
                         base_data_path: Path,
                         labeled_submissions: List[Tuple[str, int]],
                         labeler: str = "quick_add") -> Dict[str, Any]:
    """
    Quick function to add submissions by ID without creating persistent objects
    
    Args:
        dataset: TrainingDataset to add to
        base_data_path: Path to reddit data folder
        labeled_submissions: List of (submission_id, label) tuples
        labeler: Who labeled these
        
    Returns:
        Dict with addition statistics
    """
    builder = create_dataset_builder(dataset, base_data_path)
    return builder.add_batch(labeled_submissions, labeler)