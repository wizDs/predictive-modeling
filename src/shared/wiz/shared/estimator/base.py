import abc
from typing import Mapping, final, Any
import numpy as np
from wiz.evaluation import metric  # type: ignore


DoubleArray = np.typing.ArrayLike
FeatureArray = np.typing.NDArray[np.float64]


class BaseEstimator(abc.ABC):
    """Abstract base class for estimators (classifiers or regressors)."""

    @property
    @abc.abstractmethod
    def clf(self): ...

    @abc.abstractmethod
    def fit(self, features: FeatureArray, targets: DoubleArray) -> None:
        """Train the model on data X and labels y."""

    @abc.abstractmethod
    def _predict(self, features: FeatureArray) -> DoubleArray:
        """Predict outputs for input data X."""

    @final
    def predict(self, features: FeatureArray) -> DoubleArray:
        """Predict outputs for input data X."""
        return self._predict(features)

    @abc.abstractmethod
    def feature_importance(self, features: FeatureArray) -> Mapping[str, float]:
        """Predict outputs for input data X."""


class BinaryClassifier(BaseEstimator):
    """Abstract base class for classifiers."""

    @abc.abstractmethod
    def predict_proba(self, features: FeatureArray) -> DoubleArray: ...

    @final
    def score(
        self,
        features: FeatureArray,
        targets: DoubleArray,
        metric_type: metric.ClassifierMetric,
    ) -> float:
        """Calculate classification accuracy."""
        match metric_type:
            case metric.ClassifierMetric.AUC:
                proba = self.predict_proba(features)
                return metric_type.func(targets, proba)
            case _:
                prediction = self.predict(features)
                return metric_type.func(targets, prediction)


class Regressor(BaseEstimator):
    """Abstract base class for regressors."""

    def score(
        self,
        features: FeatureArray,
        targets: DoubleArray,
        metric_type: metric.RegressorMetric,
    ) -> float:
        """Calculate R² score for regression."""
        prediction = self.predict(features)
        return metric_type.func(targets, prediction)
