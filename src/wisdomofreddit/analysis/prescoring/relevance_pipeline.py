import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import json

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score, auc # Added auc here
)
import matplotlib.pyplot as plt
import seaborn as sns
from .relevance_dataset import TrainingDataset
from .submission_featurizer import SubmissionFeaturizer

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: str = "logistic_regression"  # "logistic_regression", "random_forest"
    test_size: float = 0.2 # Used only if no external test_dataset is provided
    random_state: int = 42
    cv_folds: int = 5
    scale_features: bool = True

    # Model-specific parameters
    model_params: Dict[str, Any] = None

    # Grid search parameters
    use_grid_search: bool = False
    grid_search_params: Dict[str, List] = None

    def __post_init__(self):
        if self.model_params is None:
            if self.model_type == "logistic_regression":
                self.model_params = {
                    'random_state': self.random_state,
                    'max_iter': 1000
                }
            elif self.model_type == "random_forest":
                self.model_params = {
                    'random_state': self.random_state,
                    'n_estimators': 100
                }

        if self.grid_search_params is None:
            if self.model_type == "logistic_regression":
                self.grid_search_params = {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            elif self.model_type == "random_forest":
                self.grid_search_params = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }

@dataclass
class TrainingResults:
    """Results from model training"""
    # Fields without default values first
    model_type: str
    training_config: 'TrainingConfig' # Forward reference if TrainingConfig is in the same file and defined later, or import
    train_accuracy: float
    test_accuracy: float
    cv_mean: float
    cv_std: float
    roc_auc: float
    classification_report_dict: Dict[str, Any]
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float]
    cv_scores: List[float]
    samples_trained_on: int
    samples_tested_on: int
    feature_count: int
    training_time: float
    evaluation_set_source: str

    # Fields with default values next
    pr_auc: Optional[float] = None
    y_test_true: Optional[List[int]] = None
    y_test_probabilities: Optional[List[float]] = None # Probabilities for the positive class
    best_params: Optional[Dict[str, Any]] = None
    grid_search_results: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None # Make it Optional here, __post_init__ will handle it

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class RelevanceTrainer:
    """
    Train relevance classification models on labeled submission data
    """

    def __init__(self, config: TrainingConfig = None):
        """
        Initialize trainer with configuration

        Args:
            config: Training configuration. If None, uses defaults.
        """
        self.config = config or TrainingConfig()
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_results = None
        self.feature_names = None

    def train(self,
              training_dataset: 'TrainingDataset',
              test_dataset: Optional['TrainingDataset'] = None) -> TrainingResults:
        """
        Train model on training dataset, optionally evaluating on a separate test dataset.

        Args:
            training_dataset: TrainingDataset object with labeled data for training.
            test_dataset: Optional TrainingDataset object for evaluation.
                          If None, training_dataset is split.

        Returns:
            TrainingResults with performance metrics and model info.
        """
        print(f"🚀 Starting training with {self.config.model_type}")
        start_time = datetime.now()

        # Get training data (this will be the full set used for training/fitting)
        X_train_full, y_train_full, _ = training_dataset.get_training_data()
        self.feature_names = training_dataset.get_feature_names()

        print(f"📊 Training data provided: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features")
        print(f"   Class distribution: {np.bincount(y_train_full)} (0: irrelevant, 1: relevant)")

        evaluation_source: str
        X_train_for_fit: np.ndarray
        y_train_for_fit: np.ndarray
        X_test_for_eval: np.ndarray
        y_test_for_eval: np.ndarray

        if test_dataset:
            evaluation_source = "external_dataset"
            print(f"🧪 Using provided external test dataset for evaluation.")
            if self.config.test_size != 0.2: # Default value check, or any other way to see if it was explicitly set
                print(f"   INFO: External test dataset provided. TrainingConfig.test_size ({self.config.test_size}) will be ignored.")

            X_test_external, y_test_external, _ = test_dataset.get_training_data()
            test_feature_names = test_dataset.get_feature_names()

            # Ensure feature consistency
            if self.feature_names != test_feature_names:
                if len(self.feature_names) != len(test_feature_names):
                    raise ValueError(
                        f"Feature count mismatch: training has {len(self.feature_names)} features, "
                        f"external test set has {len(test_feature_names)} features. "
                        "Ensure featurizers are compatible and have been run on both datasets."
                    )
                else: # Same length, but names differ
                    diff_idx = -1
                    for idx, (fn_train, fn_test) in enumerate(zip(self.feature_names, test_feature_names)):
                        if fn_train != fn_test:
                            diff_idx = idx
                            break
                    if diff_idx != -1:
                         raise ValueError(
                            f"Feature name mismatch at index {diff_idx}: "
                            f"training has '{self.feature_names[diff_idx]}', "
                            f"external test set has '{test_feature_names[diff_idx]}'. "
                            "Ensure featurizers are compatible and datasets are processed consistently."
                        )
                    else:
                         raise ValueError(
                            "Feature names mismatch between training and external test dataset. "
                            "Ensure featurizers are compatible and datasets are processed consistently."
                        )
            
            X_train_for_fit = X_train_full
            y_train_for_fit = y_train_full
            X_test_for_eval = X_test_external
            y_test_for_eval = y_test_external

            print(f"   Training on full provided training_dataset: {X_train_for_fit.shape[0]} samples.")
            print(f"   Evaluating on external test_dataset: {X_test_for_eval.shape[0]} samples.")
            print(f"   External Test data class distribution: {np.bincount(y_test_for_eval)} (0: irrelevant, 1: relevant)")

        else:
            evaluation_source = "internal_split"
            print(f"🔪 No external test dataset provided. Splitting training_dataset for internal testing (test_size={self.config.test_size}).")
            if X_train_full.shape[0] * self.config.test_size < 1:
                 raise ValueError(f"test_size ({self.config.test_size}) is too small for the training dataset size ({X_train_full.shape[0]}), resulting in an empty test set.")

            X_train_for_fit, X_test_for_eval, y_train_for_fit, y_test_for_eval = train_test_split(
                X_train_full, y_train_full,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y_train_full # Stratify on the full y before split
            )
            print(f"   Train split: {X_train_for_fit.shape[0]} samples.")
            print(f"   Test split: {X_test_for_eval.shape[0]} samples.")
            if X_test_for_eval.shape[0] == 0:
                print("   WARNING: Test split resulted in 0 samples. Check test_size and dataset size.")


        # Scale features if requested
        # Scaler is ALWAYS fit on the X_train_for_fit portion
        if self.config.scale_features:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_for_fit)
            X_test_scaled = self.scaler.transform(X_test_for_eval)
        else:
            X_train_scaled = X_train_for_fit
            X_test_scaled = X_test_for_eval
            self.scaler = None

        # Initialize model
        self.model = self._create_model()

        # Grid search or regular training (uses X_train_scaled, y_train_for_fit)
        grid_results_dict = None # Initialize to None
        if self.config.use_grid_search:
            print("🔍 Performing grid search...")
            best_model, grid_results_dict = self._grid_search(X_train_scaled, y_train_for_fit)
            self.model = best_model
        else:
            print("🎯 Training model...")
            self.model.fit(X_train_scaled, y_train_for_fit)

        self.is_trained = True

        # Evaluate model
        print("📈 Evaluating model...")
        results = self._evaluate_model(
            X_train_scaled, X_test_scaled, y_train_for_fit, y_test_for_eval, grid_results_dict, evaluation_source
        )

        training_time = (datetime.now() - start_time).total_seconds()
        results.training_time = training_time
        # samples_trained_on, samples_tested_on, feature_count are set in _evaluate_model

        self.training_results = results

        print(f"✅ Training completed in {training_time:.1f}s")
        print(f"   Test Accuracy (on {evaluation_source.replace('_', ' ')}): {results.test_accuracy:.3f}")
        print(f"   ROC AUC (on test data): {results.roc_auc:.3f}")
        print(f"   CV Score (on training data portion): {results.cv_mean:.3f} ± {results.cv_std:.3f}")

        return results

    def _create_model(self):
        """Create model based on configuration"""
        if self.config.model_type == "logistic_regression":
            return LogisticRegression(**self.config.model_params)
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(**self.config.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def _grid_search(self, X_train, y_train) -> Tuple[Any, Dict]:
        """Perform grid search for hyperparameter tuning"""
        base_model = self._create_model()

        grid_search = GridSearchCV(
            base_model,
            self.config.grid_search_params,
            cv=self.config.cv_folds,
            scoring='roc_auc', # Or other preferred metric
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"🏆 Best parameters from GridSearch: {grid_search.best_params_}")
        print(f"🏆 Best CV score from GridSearch (ROC AUC): {grid_search.best_score_:.3f}")

        grid_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_ # For detailed inspection if needed
        }

        return grid_search.best_estimator_, grid_results


    def _evaluate_model(self, X_train, X_test, y_train, y_test,
                        grid_search_cv_results: Optional[Dict],
                        evaluation_source: str) -> TrainingResults:
        """Comprehensive model evaluation"""

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        roc_auc_val = 0.0
        pr_auc_val = 0.0
        y_test_proba_positive_class = None

        if len(np.unique(y_test)) > 1 and X_test.shape[0] > 0: # Ensure there are samples and multiple classes
            y_test_proba_positive_class = self.model.predict_proba(X_test)[:, 1]
            roc_auc_val = roc_auc_score(y_test, y_test_proba_positive_class)
            
            precision, recall, _ = precision_recall_curve(y_test, y_test_proba_positive_class)
            pr_auc_val = auc(recall, precision)
        else:
            if X_test.shape[0] == 0:
                print("⚠️ ROC AUC and PR AUC scores cannot be calculated because the test set is empty.")
            else:
                print(f"⚠️ ROC AUC and PR AUC scores cannot be calculated because test data y_test contains only one class: {np.unique(y_test)}. Setting scores to 0.0.")
            # Initialize y_test_proba_positive_class to avoid None if y_test is single class but not empty
            if X_test.shape[0] > 0:
                 y_test_proba_positive_class = np.zeros(len(y_test)) # Or handle as appropriate

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        cv_mean_score = 0.0
        cv_std_score = 0.0
        cv_all_scores = []
        if len(y_train) >= self.config.cv_folds and len(np.unique(y_train)) > 1:
            cv_scores_obj = cross_val_score(
                self.model, X_train, y_train,
                cv=StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state),
                scoring='accuracy'
            )
            cv_mean_score = cv_scores_obj.mean()
            cv_std_score = cv_scores_obj.std()
            cv_all_scores = cv_scores_obj.tolist()
        else:
            print(f"⚠️ Cross-validation skipped: not enough samples or only one class in y_train for CV (samples: {len(y_train)}, classes: {np.unique(y_train)}).")

        class_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()
        feature_importance = self._get_feature_importance()
        
        best_params_from_grid = None
        if grid_search_cv_results:
            best_params_from_grid = grid_search_cv_results.get('best_params')

        return TrainingResults(
            model_type=self.config.model_type,
            training_config=self.config,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            cv_mean=cv_mean_score,
            cv_std=cv_std_score,
            roc_auc=roc_auc_val,
            pr_auc=pr_auc_val, # Store PR AUC
            classification_report_dict=class_report,
            confusion_matrix=conf_matrix,
            feature_importance=feature_importance,
            y_test_true=y_test.tolist() if y_test is not None else None, # Store true test labels
            y_test_probabilities=y_test_proba_positive_class.tolist() if y_test_proba_positive_class is not None else None, # Store test probabilities
            cv_scores=cv_all_scores,
            best_params=best_params_from_grid,
            grid_search_results=grid_search_cv_results,
            samples_trained_on=len(y_train),
            samples_tested_on=len(y_test) if y_test is not None else 0,
            feature_count=X_train.shape[1],
            evaluation_set_source=evaluation_source,
            training_time=0.0
        )

    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        if not self.is_trained or not self.feature_names:
            return {}

        importance_values = None
        if hasattr(self.model, 'coef_'): # Logistic Regression, LinearSVC
            if self.model.coef_.ndim > 1: # Multi-class case
                 # For binary, it's often model.coef_[0]. For simplicity, average or sum, or handle based on model.
                 importance_values = np.mean(np.abs(self.model.coef_), axis=0)
            else: # Binary classification
                importance_values = self.model.coef_[0]
        elif hasattr(self.model, 'feature_importances_'): # Tree-based models
            importance_values = self.model.feature_importances_
        else:
            return {}
        
        return dict(zip(self.feature_names, importance_values))

    def save_model(self, folder_path: Path) -> None:
        """
        Save trained model and associated components
        
        Args:
            folder_path: Folder to save model components
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = folder_path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler (if used)
        if self.scaler is not None:
            scaler_path = folder_path / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save training config
        config_path = folder_path / "training_config.json"
        with open(config_path, 'w') as f:
            # Ensure TrainingConfig is serializable (dataclasses.asdict helps)
            json.dump(asdict(self.config), f, indent=2)
        
        # Save training results
        if self.training_results:
            results_path = folder_path / "training_results.json"
            with open(results_path, 'w') as f:
                results_dict = self._serialize_training_results(self.training_results)
                json.dump(results_dict, f, indent=2)
        
        # Save feature names
        if self.feature_names:
            features_path = folder_path / "feature_names.txt"
            with open(features_path, 'w') as f:
                for name in self.feature_names:
                    f.write(f"{name}\n")
        
        print(f"✅ Model saved to: {folder_path}")

    def _serialize_training_results(self, results: TrainingResults) -> Dict[str, Any]:
        """
        Convert TrainingResults to JSON-serializable dictionary
        """
        results_dict = asdict(results) # TrainingResults is a dataclass
        
        # Convert datetime to string
        if 'timestamp' in results_dict and results_dict['timestamp']:
            results_dict['timestamp'] = results_dict['timestamp'].isoformat()
        
        # Handle numpy arrays and other non-serializable types within cv_results if present
        def convert_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, (datetime, pd.Timestamp)):
                return value.isoformat()
            elif isinstance(value, pd.DataFrame): # Example: cv_results might be a DataFrame
                return value.to_dict(orient='list')
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            else:
                return value
        
        serializable_dict = {}
        for key, value in results_dict.items():
            try:
                serializable_dict[key] = convert_value(value)
            except Exception as e:
                print(f"Warning: Could not serialize key '{key}' in TrainingResults: {e}. Storing as string.")
                serializable_dict[key] = str(value) 
        
        return serializable_dict

    def plot_results(self, save_path: Optional[Path] = None) -> None:
        """Plot training results, including ROC and Precision-Recall curves."""
        if not self.training_results:
            print("No training results to plot")
            return

        res = self.training_results
        # Increase figure size and change subplot layout to 2x3
        fig, axes = plt.subplots(2, 3, figsize=(24, 12)) # Adjusted for 6 plots
        fig.suptitle(f"Training Results: {res.model_type} (Test Set: {res.evaluation_set_source.replace('_', ' ')})", fontsize=16)

        # Flatten axes array for easier iteration if needed, or access directly
        ax_flat = axes.flatten()

        # 1. Confusion Matrix (ax_flat[0])
        conf_matrix = np.array(res.confusion_matrix)
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax_flat[0], cmap="Blues")
        ax_flat[0].set_title('Confusion Matrix (Test Set)')
        ax_flat[0].set_xlabel('Predicted Label')
        ax_flat[0].set_ylabel('True Label')

        # 2. Feature Importance (top 15) (ax_flat[1])
        importance = res.feature_importance
        if importance:
            sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_importance[:15]
            features, values = zip(*top_features)
            
            ax_flat[1].barh(range(len(features)), values, color="skyblue")
            ax_flat[1].set_yticks(range(len(features)))
            ax_flat[1].set_yticklabels(features)
            ax_flat[1].set_title('Top 15 Feature Importances')
            ax_flat[1].invert_yaxis()
        else:
            ax_flat[1].text(0.5, 0.5, "Feature importance not available.", ha='center', va='center')
            ax_flat[1].set_title('Feature Importances')

        # 3. Cross-validation scores (ax_flat[2])
        if res.cv_scores:
            ax_flat[2].hist(res.cv_scores, bins=min(10, len(res.cv_scores) if res.cv_scores else 1), alpha=0.7, color="green")
            ax_flat[2].axvline(res.cv_mean, color='red', linestyle='--', 
                             label=f'Mean: {res.cv_mean:.3f}')
            ax_flat[2].set_title(f'{res.training_config.cv_folds}-Fold CV Scores (Accuracy on Train)')
            ax_flat[2].set_xlabel('Accuracy Score')
            ax_flat[2].set_ylabel('Frequency')
            ax_flat[2].legend()
        else:
            ax_flat[2].text(0.5, 0.5, "CV scores not available.", ha='center', va='center')
            ax_flat[2].set_title('Cross-Validation Scores')
        
        # 4. ROC Curve (ax_flat[3])
        if res.y_test_true is not None and res.y_test_probabilities is not None and len(np.unique(res.y_test_true)) > 1:
            fpr, tpr, _ = roc_curve(res.y_test_true, res.y_test_probabilities)
            ax_flat[3].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {res.roc_auc:.3f})')
            ax_flat[3].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_flat[3].set_xlim([0.0, 1.0])
            ax_flat[3].set_ylim([0.0, 1.05])
            ax_flat[3].set_xlabel('False Positive Rate')
            ax_flat[3].set_ylabel('True Positive Rate')
            ax_flat[3].set_title('Receiver Operating Characteristic (Test Set)')
            ax_flat[3].legend(loc="lower right")
            ax_flat[3].grid(alpha=0.3)
        else:
            ax_flat[3].text(0.5, 0.5, "ROC curve not available.\n(Requires multi-class test data & probabilities)", ha='center', va='center', fontsize=9)
            ax_flat[3].set_title('ROC Curve (Test Set)')

        # 5. Precision-Recall Curve (ax_flat[4])
        if res.y_test_true is not None and res.y_test_probabilities is not None and len(np.unique(res.y_test_true)) > 1:
            precision, recall, _ = precision_recall_curve(res.y_test_true, res.y_test_probabilities)
            ax_flat[4].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {res.pr_auc:.3f})')
            # Plot no-skill line (baseline for PR curve)
            no_skill = len([s for s in res.y_test_true if s==1]) / len(res.y_test_true) if len(res.y_test_true) > 0 else 0
            ax_flat[4].plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label=f'No Skill (Baseline={no_skill:.2f})')
            ax_flat[4].set_xlim([0.0, 1.0])
            ax_flat[4].set_ylim([0.0, 1.05])
            ax_flat[4].set_xlabel('Recall')
            ax_flat[4].set_ylabel('Precision')
            ax_flat[4].set_title('Precision-Recall Curve (Test Set)')
            ax_flat[4].legend(loc="lower left")
            ax_flat[4].grid(alpha=0.3)
        else:
            ax_flat[4].text(0.5, 0.5, "Precision-Recall curve not available.\n(Requires multi-class test data & probabilities)", ha='center', va='center', fontsize=9)
            ax_flat[4].set_title('Precision-Recall Curve (Test Set)')

        # 6. Performance summary text (ax_flat[5])
        ax_flat[5].axis('off') 
        summary_text = f"""
        Model Performance Summary
        --------------------------
        Model Type: {res.model_type}
        Test Set: {res.evaluation_set_source.replace('_', ' ')}

        Accuracy:
        - Train: {res.train_accuracy:.3f}
        - Test: {res.test_accuracy:.3f}
        - CV Mean (Train): {res.cv_mean:.3f} (±{res.cv_std:.3f})

        Test Set Metrics:
        - ROC AUC: {res.roc_auc:.3f}
        - PR AUC: {res.pr_auc if res.pr_auc is not None else 'N/A':.3f}

        Data Split:
        - Samples Trained On: {res.samples_trained_on}
        - Samples Tested On: {res.samples_tested_on}
        - Features: {res.feature_count}

        Training Time: {res.training_time:.1f}s
        Timestamp: {res.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """
        if res.best_params:
            summary_text += f"\nBest GridSearch Params: {json.dumps(res.best_params, indent=1)}"

        ax_flat[5].text(0.05, 0.95, summary_text.strip(), transform=ax_flat[5].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.5))
        ax_flat[5].set_title('Summary', loc='left', fontsize=12, y=1.0)


        plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # Adjust layout for suptitle and bottom labels

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Results plot saved to: {save_path}")

        plt.show()

class RelevancePredictor:
    """
    Predict relevance for new submissions using trained model
    """
    
    def __init__(self, 
                 featurizer: 'SubmissionFeaturizer',
                 model: Any,
                 scaler: Optional[StandardScaler] = None,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize predictor
        
        Args:
            featurizer: SubmissionFeaturizer for feature extraction
            model: Trained sklearn model
            scaler: Fitted StandardScaler (if features were scaled during training)
            feature_names: List of feature names for validation
        """
        self.featurizer = featurizer
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
        # Validate feature compatibility
        if self.feature_names:
            featurizer_features = self.featurizer.get_feature_names()
            if featurizer_features != self.feature_names:
                raise ValueError(
                    f"Feature mismatch! Featurizer has {len(featurizer_features)} features, "
                    f"model expects {len(self.feature_names)} features"
                )
    
    @classmethod
    def load_from_folder(cls, model_folder: Path, featurizer: 'SubmissionFeaturizer') -> 'RelevancePredictor':
        """
        Load predictor from saved model folder
        
        Args:
            model_folder: Path to folder containing saved model
            featurizer: SubmissionFeaturizer to use (must match training)
            
        Returns:
            RelevancePredictor ready for predictions
        """
        model_folder = Path(model_folder)
        
        # Load model
        model_path = model_folder / "model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler (if exists)
        scaler_path = model_folder / "scaler.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        # Load feature names (if exists)
        features_path = model_folder / "feature_names.txt"
        feature_names = None
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
        
        print(f"✅ Loaded model from: {model_folder}")
        
        return cls(featurizer, model, scaler, feature_names)
    
    def predict_single(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict relevance for a single submission
        
        Args:
            submission_data: JSON submission data
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        feature_result = self.featurizer.extract_features(submission_data)
        
        # Prepare feature vector
        X = feature_result.feature_vector.reshape(1, -1)
        
        # Scale if scaler was used during training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        return {
            'submission_id': submission_data.get('submission_id', 'unknown'),
            'predicted_label': int(prediction),
            'predicted_class': 'relevant' if prediction == 1 else 'irrelevant',
            'probability_irrelevant': float(probabilities[0]),
            'probability_relevant': float(probabilities[1]),
            'confidence': float(max(probabilities)),
            'features': feature_result.features,
            'feature_metadata': feature_result.metadata
        }
    
    def predict_batch(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict relevance for multiple submissions
        
        Args:
            submissions: List of submission JSON data
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"🔮 Predicting relevance for {len(submissions)} submissions...")
        
        for i, submission_data in enumerate(submissions):
            try:
                result = self.predict_single(submission_data)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(submissions)} submissions")
                    
            except Exception as e:
                print(f"   Error processing submission {submission_data.get('submission_id', i)}: {e}")
                # Add error result
                results.append({
                    'submission_id': submission_data.get('submission_id', f'error_{i}'),
                    'predicted_label': 0,
                    'predicted_class': 'error',
                    'probability_irrelevant': 0.5,
                    'probability_relevant': 0.5,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        print(f"✅ Completed predictions for {len(results)} submissions")
        return results
    
    def filter_relevant_submissions(self, 
                                  submissions: List[Dict[str, Any]], 
                                  min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """
        Filter submissions to only relevant ones above confidence threshold
        
        Args:
            submissions: List of submission JSON data
            min_confidence: Minimum confidence threshold for relevance
            
        Returns:
            List of submissions predicted as relevant with high confidence
        """
        predictions = self.predict_batch(submissions)
        
        relevant_submissions = []
        for submission, prediction in zip(submissions, predictions):
            if (prediction['predicted_label'] == 1 and 
                prediction['confidence'] >= min_confidence):
                
                # Add prediction metadata to submission
                submission_with_prediction = submission.copy()
                submission_with_prediction['relevance_prediction'] = prediction
                relevant_submissions.append(submission_with_prediction)
        
        print(f"🎯 Filtered {len(relevant_submissions)}/{len(submissions)} submissions as relevant "
              f"(confidence >= {min_confidence})")
        
        return relevant_submissions
    
    def get_prediction_summary(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics of predictions"""
        if not predictions:
            return {}
        
        relevant_count = sum(1 for p in predictions if p['predicted_label'] == 1)
        confidences = [p['confidence'] for p in predictions if 'confidence' in p]
        
        return {
            'total_predictions': len(predictions),
            'relevant_count': relevant_count,
            'irrelevant_count': len(predictions) - relevant_count,
            'relevance_rate': relevant_count / len(predictions),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'min_confidence': np.min(confidences) if confidences else 0.0,
            'max_confidence': np.max(confidences) if confidences else 0.0
        }

