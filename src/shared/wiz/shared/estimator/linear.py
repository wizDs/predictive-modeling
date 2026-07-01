import logging
from sklearn import linear_model
import numpy as np
from .estimator import Regressor
from wiz.interface import estimator_interface
from wiz.shared.estimator import FeatureArray, DoubleArray, RegressionCoefficients
from typing import Mapping


logger = logging.getLogger(__name__)


class LinearModel(Regressor):

    def __init__(self, estimator: estimator_interface.LinearRegression, /) -> None:
        super().__init__()
        self.clf = linear_model.LinearRegression(
            **estimator.model_dump(exclude=["estimator_type"])
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


