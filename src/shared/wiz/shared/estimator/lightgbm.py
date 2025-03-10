import lightgbm as lgb
import numpy as np
from .estimator import BinaryClassifier, Regressor
from wiz.interface import estimator_interface


class LGBMClassifier(BinaryClassifier):

    def __init__(self, estimator: estimator_interface.LGBMClassifier, /) -> None:
        super().__init__()
        self.clf = lgb.LGBMClassifier(
            **estimator.model_dump(exclude=["estimator_type"])
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.clf.fit(features, targets)

    def _predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(features)

    def feature_importance(self):
        # https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
        return None  # self.clf_booster.get_score(importance_type="gain")


class LGBMRegressor(Regressor):

    def __init__(self, estimator: estimator_interface.LGBMRegressor, /) -> None:
        super().__init__()
        self.clf = lgb.LGBMRegressor(**estimator.model_dump(exclude=["estimator_type"]))

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.clf.fit(features, targets)

    def _predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict(features)

    def feature_importance(self):
        # https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
        return None  # self.clf_booster.get_score(importance_type="gain")
