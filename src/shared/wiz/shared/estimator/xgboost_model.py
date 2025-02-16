import xgboost
import numpy as np
from .base import BinaryClassifier
from wiz.interface import modeling_interface


class XGBoostClassifier(BinaryClassifier):

    def __init__(self, estimator: modeling_interface.EstimatorInterface) -> None:
        super().__init__()
        clf = xgboost.XGBClassifier(estimator.model_dump())

    def fit(self, features: np.ndarray) -> None:
        
        
    def _predict(self, features):
        return self.clf