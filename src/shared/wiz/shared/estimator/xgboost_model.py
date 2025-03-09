import xgboost
import numpy as np
from .estimator import BinaryClassifier, Regressor
from wiz.interface import estimator_interface


class XGBoostClassifier(BinaryClassifier):

    def __init__(self, estimator: estimator_interface.XGBoostClassifier) -> None:
        super().__init__()
        self.clf = xgboost.XGBClassifier(
            **estimator.model_dump(exclude=["estimator_type"])
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.clf.fit(features, targets)
        self.clf_booster = self.clf.get_booster()

    def _predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(features)

    def feature_importance(self):
        # https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
        return self.clf_booster.get_score(importance_type="gain")


class XGBoostRegressor(Regressor):

    def __init__(self, estimator: estimator_interface.XGBoostRegressor) -> None:
        super().__init__()
        self.clf = xgboost.XGBRegressor(
            **estimator.model_dump(exclude=["estimator_type"])
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.clf.fit(features, targets)
        self.clf_booster = self.clf.get_booster()

    def _predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(features)

    def feature_importance(self):
        # https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
        return self.clf_booster.get_score(importance_type="gain")
