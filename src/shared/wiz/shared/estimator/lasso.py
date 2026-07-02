from sklearn import linear_model
import numpy as np
from wiz.shared.estimator.estimator import (
    Regressor,
    FeatureArray,
    DoubleArray,
    RegressionCoefficients,
)
from wiz.interface import estimator_interface
from typing import Mapping


class LassoModel(Regressor):

    def __init__(self, estimator: estimator_interface.LassoModel, /) -> None:
        super().__init__()
        self.clf = linear_model.Lasso(
            **estimator.model_dump(exclude={"estimator_type"})
        )

    def fit(self, features: FeatureArray, targets: DoubleArray) -> None:
        self.clf.fit(features, targets)

    def _predict(self, features: FeatureArray) -> DoubleArray:
        return self.clf.predict(features)

    def feature_importance(self, features: FeatureArray) -> RegressionCoefficients:
        assert isinstance(self.clf.intercept_, float)
        return {
            "intercept": self.clf.intercept_,
            "coefficients": [float(x) for x in self.clf.coef_[0]],
        }


