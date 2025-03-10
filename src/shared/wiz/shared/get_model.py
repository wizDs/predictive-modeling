from wiz.shared import estimator
from wiz.shared.preprocessor import preprocessor
from wiz.interface import estimator_interface, preproc_interface, target_interface
from wiz.shared.target_transformer import target_transformer


def model_from_type(
    estimator_type: estimator_interface.EstimatorType, /
) -> estimator.BaseEstimator:
    match estimator_type:
        case estimator_interface.XGBoostClassifier():
            return estimator.XGBoostClassifier(estimator_type)
        case estimator_interface.XGBoostRegressor():
            return estimator.XGBoostRegressor(estimator_type)
        case estimator_interface.LinearRegression():
            return estimator.LinearModel(estimator_type)
        case estimator_interface.LassoModel():
            return estimator.LassoModel(estimator_type)
        case estimator_interface.KNeighborsRegressor():
            return estimator.KNeighborsRegressor(estimator_type)
        case estimator_interface.LGBMRegressor():
            return estimator.LGBMRegressor(estimator_type)
        case estimator_interface.LGBMClassifier():
            return estimator.LGBMClassifier(estimator_type)


def preprocessor_from_type(preprocessor_type: preproc_interface.PreProcInterface):
    return preprocessor.DefaultPreProcessor(preprocessor_type)


def target_from_type(target_type: target_interface.TargetInterface):
    match target_type:
        case target_interface.DummyTransformer():
            return target_transformer.DummyTransformer()
        case target_interface.LogTransformer():
            return target_transformer.LogTransformer()
        case target_interface.PowerTransformer():
            return target_transformer.PowerTransformer(target_type)
