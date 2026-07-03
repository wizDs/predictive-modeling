from typing import Mapping
import numpy as np
import lightgbm as lgb
from wiz.shared.estimator import estimator
from wiz.shared.estimator.estimator import FeatureArray, DoubleArray
from wiz.interface import estimator_interface


class LGBMClassifier(estimator.BinaryClassifier):

    def __init__(self, estimator: estimator_interface.LGBMClassifier, /) -> None:
        super().__init__()
        self.clf = lgb.LGBMClassifier(
            **estimator.model_dump(exclude={"estimator_type"})
        )

    def fit(self, features: FeatureArray, targets: DoubleArray) -> None:
        self.clf.fit(features, targets)

    def _predict(self, features: FeatureArray) -> DoubleArray:
        return self.clf.predict(features)

    def predict_proba(self, features: FeatureArray) -> DoubleArray:
        return self.clf.predict_proba(features)

    def feature_importance(self, features: FeatureArray) -> Mapping[str, float]:
        # https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
        return {}  # self.clf_booster.get_score(importance_type="gain")


class LGBMRegressor(estimator.Regressor):

    def __init__(self, estimator: estimator_interface.LGBMRegressor, /) -> None:
        super().__init__()
        self.clf = lgb.LGBMRegressor(**estimator.model_dump(exclude={"estimator_type"}))

    def fit(self, features: FeatureArray, targets: DoubleArray) -> None:
        self.clf.fit(features, targets)

    def _predict(self, features: FeatureArray) -> DoubleArray:
        return self.clf.predict(features)

    def feature_importance(self):
        # https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
        return None  # self.clf_booster.get_score(importance_type="gain")
