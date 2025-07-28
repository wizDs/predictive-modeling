from sklearn import neighbors
import numpy as np
from wiz.shared.estimator import Regressor
from wiz.interface import estimator_interface
from wiz.interface.feature_array import FeatureArray
from typing import Mapping
from wiz.interface.feature_array import DoubleArray


class KNeighborsRegressor(Regressor):

    def __init__(self, estimator: estimator_interface.KNeighborsRegressor, /) -> None:
        super().__init__()
        self.clf = neighbors.KNeighborsRegressor(
            **estimator.model_dump(exclude=["estimator_type"])
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.clf.fit(features, targets)

    def _predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(features)

    def predict_proba(self, features: FeatureArray) -> DoubleArray:
        return self.clf.predict_proba(features)

    def feature_importance(self, features: FeatureArray) -> Mapping[str, float] | None:
        return None  # self.clf.intercept_, self.clf.coef_[0]
