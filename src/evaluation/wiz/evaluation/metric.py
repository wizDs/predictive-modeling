import enum
import sklearn
from typing import TypeAlias


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


class RegressorMetric(enum.StrEnum):
    """Common regression evaluation metrics."""

    R2 = "r2"
    """Measures how well predictions fit actual values (higher is better)."""
    RMSE = "rmse"
    """Root mean squared error; penalizes larger errors more heavily."""
    MAPE = "mape"
    """Mean absolute percentage error; average pct difference between predictions and actuals."""
    MPE = "mpe"
    """Mean percentage error; like MAPE but retains the sign (bias indicator)."""
    MEAN_ERROR = "mean_error"
    """Average difference between predicted and actual values (can be negative)."""
    MEAN_ABS_ERROR = "mean_abs_error"
    """Average absolute difference between predictions and actual values."""


EvaluationMetric: TypeAlias = RegressorMetric | ClassifierMetric
