"""
Strategy Builder: Random Forest → Feature Selection → Decision Tree → Rules (Fases 5-9).

Pipeline:
1. Load labeled dataset (features + labels)
2. Temporal split 70/30 (train/test)
3. Train Random Forest (feature selection)
4. Select top 3-5 features
5. Train Decision Tree (max_depth=3/4)
6. Extract rules from tree leaves
7. Filter valid strategies (P(Label=1)>0.5, n_samples>=30)
8. Return structured strategy definitions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json

# ML imports (lazy to avoid hard dependency)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    DecisionTreeClassifier = None


@dataclass
class StrategyRule:
    """Single strategy extracted from Decision Tree leaf."""
    rule_id: int
    conditions: List[str]  # e.g., ["RSI_14 > 55", "MACD_line > 0"]
    n_samples: int
    p_label_1: float  # Probability of TP
    win_rate_train: Optional[float] = None
    metrics_train: Optional[Dict] = None
    metrics_test: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class StrategyBuilder:
    """Build strategies from data using RF + Tree pipeline."""
    
    def __init__(self, feature_cols: Optional[List[str]] = None,
                 top_n_features: int = 5,
                 max_depth: int = 4,
                 min_samples_leaf: int = 30,
                 random_state: int = 42):
        """
        Args:
            feature_cols: List of feature column names (if None, auto-detect)
            top_n_features: Number of top features to keep from RF
            max_depth: Max depth for Decision Tree
            min_samples_leaf: Minimum samples per leaf (for strategy validity)
            random_state: Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for StrategyBuilder")
        
        self.feature_cols = feature_cols
        self.top_n_features = top_n_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.rf_model = None
        self.tree_model = None
        self.selected_features = None
        self.strategies: List[StrategyRule] = []
    
    def prepare_data(self, df: pd.DataFrame, label_col: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare X, y with temporal split 70/30.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Auto-detect features if not specified
        if self.feature_cols is None:
            exclude_cols = ['label', 'entry_idx', 'Datetime', 'label']
            all_cols = [c for c in df.columns if c not in exclude_cols]
            # Filter numeric only
            self.feature_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        
        # Handle missing values
        df_clean = df.dropna(subset=self.feature_cols + [label_col])
        
        X = df_clean[self.feature_cols].values
        y = df_clean[label_col].values
        
        # Temporal split 70/30 (chronological, no shuffling)
        n_train = int(len(X) * 0.7)
        
        X_train = X[:n_train]
        X_test = X[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Train Random Forest for feature importance.
        
        Returns:
            Feature importances array (same order as self.feature_cols)
        """
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=50,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        importances = self.rf_model.feature_importances_
        return importances
    
    def select_features(self, importances: np.ndarray) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Select top N features by importance.
        
        Returns:
            X_train_reduced, X_test_reduced, selected_feature_names, selected_indices
        """
        # Get indices of top N features
        top_indices = np.argsort(importances)[-self.top_n_features:][::-1]
        selected_indices = top_indices.tolist()
        selected_names = [self.feature_cols[i] for i in selected_indices]
        
        # We'll store for later use
        self.selected_feature_indices = selected_indices
        self.selected_features = selected_names
        
        return selected_indices, selected_names
    
    def train_decision_tree(self, X_train: np.ndarray, y_train: np.ndarray,
                            selected_indices: List[int]) -> DecisionTreeClassifier:
        """
        Train small Decision Tree on selected features only.
        """
        # Select features
        X_train_sel = X_train[:, selected_indices]
        
        self.tree_model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion='gini',
            random_state=self.random_state
        )
        self.tree_model.fit(X_train_sel, y_train)
        
        return self.tree_model
    
    def extract_rules(self, feature_names: List[str]) -> List[StrategyRule]:
        """
        Extract rules from Decision Tree leaves.
        
        Each leaf becomes a strategy IF:
          - P(Label=1) > 0.5
          - n_samples >= min_samples_leaf
        
        Returns:
            List of valid StrategyRule objects
        """
        tree = self.tree_model
        n_features = len(feature_names)
        
        # Get tree structure
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value  # Class counts per node
        
        strategies = []
        
        def traverse(node_id, condition_list, n_samples):
            """DFS traversal of tree."""
            # Check if leaf
            if children_left[node_id] == children_right[node_id]:
                # Leaf node
                if n_samples >= self.min_samples_leaf:
                    # Calculate P(Label=1)
                    class_counts = value[node_id][0]
                    total = class_counts.sum()
                    p_label_1 = class_counts[1] / total if total > 0 else 0
                    
                    if p_label_1 > 0.5:
                        # Valid strategy
                        strategy = StrategyRule(
                            rule_id=len(strategies),
                            conditions=condition_list.copy(),
                            n_samples=int(n_samples),
                            p_label_1=float(p_label_1)
                        )
                        strategies.append(strategy)
                return
            
            # Non-leaf: split on feature
            split_feat = feature[node_id]
            split_thresh = threshold[node_id]
            feat_name = feature_names[split_feat]
            
            # Left child (feature <= threshold)
            traverse(children_left[node_id],
                     condition_list + [f"{feat_name} <= {split_thresh:.3f}"],
                     value[node_id].sum())
            
            # Right child (feature > threshold)
            traverse(children_right[node_id],
                     condition_list + [f"{feat_name} > {split_thresh:.3f}"],
                     value[node_id].sum())
        
        # Start traversal from root
        traverse(0, [], value[0].sum())
        
        self.strategies = strategies
        return strategies
    
    def fit(self, df: pd.DataFrame, label_col: str = 'label') -> List[StrategyRule]:
        """
        Full pipeline: fit RF → select features → fit Tree → extract rules.
        
        Args:
            df: DataFrame with features + label column
            label_col: Name of label column
        
        Returns:
            List of valid strategy rules
        """
        # Step 1: Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, label_col)
        
        # Step 2: Train Random Forest
        importances = self.train_random_forest(X_train, y_train)
        
        # Step 3: Select top features
        selected_indices, selected_names = self.select_features(importances)
        
        # Step 4: Train Decision Tree
        self.train_decision_tree(X_train, y_train, selected_indices)
        
        # Step 5: Extract rules
        rules = self.extract_rules(selected_names)
        
        print(f"Extracted {len(rules)} valid strategies from {len(df)} samples")
        for r in rules:
            print(f"  - Rule {r.rule_id}: {r.conditions} (n={r.n_samples}, P(TP)={r.p_label_1:.3f})")
        
        return rules
    
    def export_strategies(self, filepath: str):
        """Export valid strategies to JSON."""
        with open(filepath, 'w') as f:
            json.dump([s.to_dict() for s in self.strategies], f, indent=2)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importances from Random Forest."""
        if self.rf_model is None:
            return {}
        return dict(zip(self.feature_cols, self.rf_model.feature_importances_))