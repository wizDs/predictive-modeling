import xgboost
from .base import BinaryClassifier


class XGBoostClassifier(BinaryClassifier):

    def __init__(self) -> None:
        # super().__init__()
        clf = xgboost.XGBClassifier()
