import enum
from typing import Callable, TypeAlias, assert_never
import numpy as np
from sklearn import metrics  # type: ignore


class ClassifierMetric(enum.StrEnum):
    """Common classifier evaluation metrics."""

    ACCURACY = "accuracy"
    """Proportion of correctly classified instances among the total instances."""
    PRECISION = "precision"
    """Proportion of true positive predictions among all positive predictions (TP / (TP + FP))."""
    RECALL = "recall"
    """Proportion of actual positive instances that were correctly identified (TP / (TP + FN))."""
    AUC = "auc"
    """Area under the ROC curve, representing the model's ability to distinguish between classes."""

    @property
    def func(self: "ClassifierMetric") -> Callable[..., float]:
        match self:
            case ClassifierMetric.ACCURACY:
                return metrics.accuracy_score
            case ClassifierMetric.PRECISION:
                return metrics.precision_score
            case ClassifierMetric.RECALL:
                return metrics.recall_score
            case ClassifierMetric.AUC:
                return self._auc
            case _:
                assert_never(self)

    @staticmethod
    def _auc(y_true: np.ndarray, y_proba: np.ndarray, pos_label: int) -> float:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_proba, pos_label=pos_label)
        return metrics.auc(fpr, tpr)


class RegressorMetric(enum.StrEnum):
    """Common regression evaluation metrics."""

    R2 = "r2"
    """Measures how well predictions fit actual values (higher is better)."""
    MAE = "mae"
    """Mean absolute error; average absolute difference between predictions and actual values."""
    RMSE = "rmse"
    """Root mean squared error; penalizes larger errors more heavily."""
    MAPE = "mape"
    """Mean absolute percentage error; average pct difference between predictions and actuals."""
    MEAN_ERROR = "mean_error"
    """Average difference between predicted and actual values (can be negative)."""
    MEDIAN_ABS_ERROR = "median_abs_error"
    """Median absolute difference between predictions and actual values."""

    @property
    def func(self: "RegressorMetric") -> Callable[..., float]:
        match self:
            case RegressorMetric.R2:
                return metrics.r2_score
            case RegressorMetric.MAE:
                return metrics.mean_absolute_error
            case RegressorMetric.RMSE:
                return metrics.root_mean_squared_error
            case RegressorMetric.MAPE:
                return metrics.mean_absolute_percentage_error
            case RegressorMetric.MEAN_ERROR:
                return self._mean_error
            case RegressorMetric.MEDIAN_ABS_ERROR:
                return metrics.median_absolute_error
            case _:
                assert_never(self)

    @staticmethod
    def _mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_pred - y_true
        return float(np.mean(error))


EvaluationMetricType: TypeAlias = RegressorMetric | ClassifierMetric
