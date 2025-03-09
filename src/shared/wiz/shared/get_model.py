from wiz.shared import estimator
from wiz.shared.preprocessor import preprocessor
from wiz.interface import estimator_interface, preproc_interface


def model_from_type(
    estimator_type: estimator_interface.EstimatorType, /
) -> estimator.BaseEstimator:
    match estimator_type:
        case estimator_interface.XGBoostClassifier():
            return estimator.XGBoostClassifier(estimator_type)
        case estimator_interface.XGBoostRegressor():
            return estimator.XGBoostRegressor(estimator_type)


def preprocessor_from_type(preprocessor_type: preproc_interface.DefaultPreProcessor):
    return preprocessor.DefaultPreProcessor(preprocessor_type)
