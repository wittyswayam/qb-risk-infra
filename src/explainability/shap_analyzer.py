"""SHAP-based explainability for strategy feature contributions.

Fits a gradient-boosted tree classifier on feature-labelled signal data and
computes SHAP values to explain which features drove each signal. The output
is intended as a diagnostic tool for understanding strategy behaviour, not
as a live trading signal.

Dependency: scikit-learn (GradientBoostingClassifier). SHAP library is
optional but strongly recommended for full analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Fits a surrogate model on strategy signals and computes feature importance.

    Workflow:
    1. Align feature matrix X with strategy signal labels y.
    2. Fit a GradientBoostingClassifier as a surrogate model.
    3. Compute permutation importance and (if SHAP is installed) SHAP values.
    4. Return an interpretability report.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        cv_folds: Number of folds for cross-validated accuracy reporting.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv_folds = cv_folds
        self.random_state = random_state
        self._pipeline: Optional[Pipeline] = None
        self._feature_names: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        signals: pd.Series,
    ) -> dict[str, Any]:
        """Fit surrogate model on features aligned to strategy signals.

        Args:
            features: Feature matrix from FeatureBuilder.build().
            signals: Strategy direction series (+1, 0, -1).

        Returns:
            Training diagnostics (accuracy, CV scores).
        """
        X, y = self._align(features, signals)
        if len(X) < 50:
            raise ValueError(
                f"Need at least 50 aligned samples to fit; got {len(X)}."
            )

        self._feature_names = list(X.columns)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                learning_rate=0.05,
                subsample=0.8,
            )),
        ])

        # Cross-validated accuracy for diagnostics
        cv_scores = cross_val_score(
            self._pipeline, X.values, y.values, cv=self.cv_folds, scoring="accuracy"
        )

        self._pipeline.fit(X.values, y.values)
        train_acc = self._pipeline.score(X.values, y.values)

        logger.info(
            "SHAPAnalyzer fit: train_acc=%.4f cv_acc=%.4f+/-%.4f n_samples=%d",
            train_acc,
            cv_scores.mean(),
            cv_scores.std(),
            len(X),
        )

        return {
            "train_accuracy": round(float(train_acc), 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "n_samples": len(X),
            "n_features": len(self._feature_names),
        }

    def feature_importance(
        self, features: pd.DataFrame, signals: pd.Series
    ) -> pd.DataFrame:
        """Compute permutation importance for all features.

        Permutation importance measures how much model performance degrades
        when a feature column is randomly shuffled. It is model-agnostic
        and more reliable than impurity-based importances for correlated features.

        Args:
            features: Feature matrix.
            signals: Aligned signal labels.

        Returns:
            DataFrame with columns: feature, importance_mean, importance_std.
            Sorted descending by importance_mean.
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() before feature_importance().")

        X, y = self._align(features, signals)
        result = permutation_importance(
            self._pipeline,
            X.values,
            y.values,
            n_repeats=20,
            random_state=self.random_state,
            n_jobs=-1,
        )

        importance_df = pd.DataFrame({
            "feature": self._feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        return importance_df

    def shap_values(
        self, features: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Compute SHAP values using the TreeExplainer if SHAP is installed.

        Returns None if shap is not installed, with a logged warning.

        Args:
            features: Feature matrix (aligned to training features).

        Returns:
            Array of SHAP values shape (n_samples, n_features) or None.
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() before shap_values().")

        try:
            import shap  # type: ignore
        except ImportError:
            logger.warning(
                "shap not installed. Install with: pip install shap. "
                "Falling back to permutation importance."
            )
            return None

        clf = self._pipeline.named_steps["clf"]
        scaler = self._pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(features[self._feature_names].values)

        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(X_scaled)

        # For multi-class, take the positive signal (class index 1 or 2 depending on label encoding)
        if isinstance(values, list):
            values = values[0]  # class 0 by default; adjust for multi-class

        logger.info("SHAP values computed for %d samples.", len(features))
        return values

    def interpretability_report(
        self,
        features: pd.DataFrame,
        signals: pd.Series,
    ) -> dict[str, Any]:
        """Generate a full interpretability report.

        Combines feature importance, SHAP values (if available), and
        signal distribution statistics.

        Args:
            features: Feature matrix.
            signals: Strategy signal series.

        Returns:
            Dictionary with report sections.
        """
        importance_df = self.feature_importance(features, signals)
        shap_vals = self.shap_values(features)

        signal_dist = signals.value_counts(normalize=True).to_dict()
        top_features = importance_df.head(10)["feature"].tolist()

        report: dict[str, Any] = {
            "signal_distribution": {str(k): round(v, 4) for k, v in signal_dist.items()},
            "top_features": top_features,
            "feature_importance": importance_df.to_dict(orient="records"),
            "shap_available": shap_vals is not None,
        }

        if shap_vals is not None:
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_ranking = pd.DataFrame({
                "feature": self._feature_names,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False)
            report["shap_feature_ranking"] = shap_ranking.to_dict(orient="records")

        return report

    @staticmethod
    def _align(features: pd.DataFrame, signals: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Align features and signals on their common index, drop NaN."""
        combined = features.join(signals.rename("__signal__"), how="inner")
        combined = combined.dropna()
        X = combined.drop(columns=["__signal__"])
        y = combined["__signal__"].astype(int)
        return X, y
