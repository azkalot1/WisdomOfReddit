from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

@dataclass
class LabeledFeaturizedSample:
    """Result of feature extraction"""
    submission_id: str
    subreddit: str
    date: str
    features: Dict[str, float]
    feature_vector: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]
    label: int

@dataclass
class FeaturizedSample:
    """Result of feature extraction"""
    submission_id: str
    subreddit: str
    date: str
    features: Dict[str, float]
    feature_vector: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]

@dataclass
class LabeledSubmission:
    """Single labeled submission for training"""
    submission_id: str
    subreddit: str
    date: str
    submission_data: Dict[str, Any]  # Original JSON submission
    label: int  # 0 or 1 (irrelevant or relevant)
    labeler: str  # Who labeled it
    confidence: float  # Labeler confidence (1-5)
    timestamp: datetime  # When it was labeled
    notes: str = ""  # Optional notes

@dataclass
class TrainingDatasetMetadata:
    """Metadata about the training dataset"""
    dataset_name: str
    creation_date: datetime
    total_samples: int
    positive_samples: int
    negative_samples: int
    labelers: List[str]
    featurizer_hash: str  # Hash of featurizer configuration
    feature_count: int
    feature_names: List[str]
    data_sources: List[str]  # Where submissions came from
    notes: str = ""