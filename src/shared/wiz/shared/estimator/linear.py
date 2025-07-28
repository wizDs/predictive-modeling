from sklearn import linear_model
import numpy as np
from .estimator import Regressor
from wiz.interface import estimator_interface
from wiz.interface.feature_array import FeatureArray
from typing import Mapping


class LinearModel(Regressor):

    def __init__(self, estimator: estimator_interface.LinearRegression, /) -> None:
        super().__init__()
        self.clf = linear_model.LinearRegression(
            **estimator.model_dump(exclude=["estimator_type"])
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.clf.fit(features, targets)

    def _predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(features)

    def feature_importance(self, features: FeatureArray) -> Mapping[str, float]:
        return self.clf.intercept_, self.clf.coef_[0]
