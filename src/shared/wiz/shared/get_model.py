from wiz.shared import estimator
from wiz.interface import estimator_interface


def model_from_type(
    estimator_type: estimator_interface.EstimatorType, /
) -> estimator.BaseEstimator:
    match estimator_type:
        case estimator_interface.XGBoostClassifier():
            return estimator.XGBoostClassifier(estimator=estimator_type)
        case estimator_interface.XGBoostRegressor():
            return estimator.XGBoostRegressor(estimator=estimator_type)
