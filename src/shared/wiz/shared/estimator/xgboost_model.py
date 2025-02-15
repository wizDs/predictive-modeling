import xgboost as xgb  # type: ignore
from .base import BinaryClassifier


class XGBoostClassifier(BinaryClassifier):

    def __init__(self):
        super().__init__()
        xgb.XGBClassifier()
