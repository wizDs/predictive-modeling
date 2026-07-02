import abc
from collections.abc import Sequence
from typing import Mapping, TypeAlias, TypedDict, final, overload
import numpy as np
from wiz.evaluation import metric  # type: ignore


DoubleArray = np.typing.ArrayLike
FeatureArray = np.typing.NDArray[np.float64]

class RegressionCoefficients(TypedDict):
    intercept: float
    coeficients: Sequence[float]

FeatureImportance: TypeAlias = Mapping[str, float] | Mapping[str, Sequence[float]]

class BaseEstimator(abc.ABC):
    """Abstract base class for estimators (classifiers or regressors)."""

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
    def feature_importance(self, features: FeatureArray) -> RegressionCoefficients | FeatureImportance | None:
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

    @abc.abstractmethod
    def feature_importance(self, features: FeatureArray) -> FeatureImportance | None:
        """Predict outputs for input data X."""


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

    @abc.abstractmethod
    def feature_importance(self, features: FeatureArray) -> RegressionCoefficients | None:
        """Predict outputs for input data X."""
