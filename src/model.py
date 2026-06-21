"""
LightGBM Model Training and Evaluation.
Stubs for future implementation - see roadmap in README.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """LightGBM configuration parameters."""
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1


def prepare_features_for_lightgbm(df: pd.DataFrame,
                                   feature_cols: List[str],
                                   label_col: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and labels for LightGBM training.

    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        label_col: Target column name

    Returns:
        (X, y) tuple
    """
    # TODO: Implement feature preparation
    # - Handle missing values
    # - Categorical feature encoding
    # - Feature scaling if needed
    # - Train/validation split (time-aware)
    raise NotImplementedError("Feature preparation - pending implementation")


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    config: ModelConfig,
                    categorical_features: Optional[List[str]] = None) -> 'lgb.Booster':
    """
    Train LightGBM model with early stopping.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: ModelConfig instance
        categorical_features: List of categorical column names

    Returns:
        Trained LightGBM Booster

    TODO: Implement with:
    - Early stopping
    - Class weights for imbalanced labels
    - Optuna hyperparameter optimization
    - Feature importance extraction
    """
    # TODO: Import lightgbm and implement training
    raise NotImplementedError("LightGBM training - pending implementation")


def walk_forward_validation(df: pd.DataFrame,
                             feature_cols: List[str],
                             label_col: str,
                             train_window: int = 252,
                             test_window: int = 63,
                             step: int = 21) -> List[Dict]:
    """
    Walk-forward validation for time series.

    Args:
        df: Full dataset sorted by time
        feature_cols: Feature columns
        label_col: Target column
        train_window: Training window size (bars)
        test_window: Test window size (bars)
        step: Step size between folds

    Returns:
        List of fold results with metrics

    TODO: Implement walk-forward splits
    """
    raise NotImplementedError("Walk-forward validation - pending implementation")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Evaluate model with classification metrics.

    Returns dict with: accuracy, precision, recall, f1, auc, confusion_matrix
    """
    # TODO: Implement evaluation metrics
    raise NotImplementedError("Model evaluation - pending implementation")


def predict_with_threshold(model, X: pd.DataFrame,
                            threshold: float = 0.5) -> np.ndarray:
    """Generate binary predictions with custom threshold."""
    # TODO: Implement prediction with threshold
    raise NotImplementedError("Prediction - pending implementation")


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Extract and format feature importance from trained model."""
    # TODO: Implement feature importance extraction
    raise NotImplementedError("Feature importance - pending implementation")