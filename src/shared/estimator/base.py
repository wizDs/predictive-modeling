import abc
from typing import final
import numpy as np


class BaseEstimator(abc.ABC):
    """Abstract base class for estimators (classifiers or regressors)."""

    @abc.abstractmethod
    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Train the model on data X and labels y."""

    @abc.abstractmethod
    def _predict(self, features: np.ndarray) -> np.ndarray:
        """Predict outputs for input data X."""

    @final
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict outputs for input data X."""
        return self._predict(features)

    @abc.abstractmethod
    def feature_importance(self, features: np.ndarray) -> np.ndarray:
        """Predict outputs for input data X."""

    @abc.abstractmethod
    def score(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Evaluate model performance"""


class Classifier(BaseEstimator):
    """Abstract base class for classifiers."""

    def score(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Calculate classification accuracy."""
        predictions = self.predict(features)
        return np.mean(predictions == targets)  # Accuracy metric


class Regressor(BaseEstimator):
    """Abstract base class for regressors."""

    def score(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Calculate R² score for regression."""
        predictions = self.predict(features)
        ss_total = np.sum((targets - np.mean(targets)) ** 2)
        ss_residual = np.sum((targets - predictions) ** 2)
        return 1 - (ss_residual / ss_total)  # R² score
