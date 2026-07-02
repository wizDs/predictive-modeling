from sklearn import neighbors
from wiz.shared.estimator.estimator import (
    Regressor,
    FeatureArray,
    DoubleArray,
    RegressionCoefficients,
)
from wiz.interface import estimator_interface


class KNeighborsRegressor(Regressor):

    def __init__(self, estimator: estimator_interface.KNeighborsRegressor, /) -> None:
        super().__init__()
        self.clf = neighbors.KNeighborsRegressor(
            **estimator.model_dump(exclude={"estimator_type"})
        )

    def fit(self, features: FeatureArray, targets: DoubleArray) -> None:
        self.clf.fit(features, targets)

    def _predict(self, features: FeatureArray) -> DoubleArray:
        return self.clf.predict(features)

    def predict_proba(self, features: FeatureArray) -> DoubleArray:
        return self.clf.predict_proba(features)

    def feature_importance(self, features: FeatureArray) -> RegressionCoefficients | None:
        return None  # self.clf.intercept_, self.clf.coef_[0]
