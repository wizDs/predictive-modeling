from wiz.shared import estimator
from wiz.interface import modeling_interface


def model_from_type(
    estimator_type: modeling_interface.EstimatorType,
) -> type[estimator.BaseEstimator]:
    match estimator_type:
        case modeling_interface.XGBoostClassifier:
            return estimator.XGBoostClassifier
