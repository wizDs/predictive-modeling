from typing import Callable, Literal, Annotated, Optional, TypeAlias
import numpy as np
import pydantic
import xgboost


class XGBoostClassifier(pydantic.BaseModel):
    """https://xgboost.readthedocs.io/en/stable/parameter.html"""

    # 'objective': 'multi:softprob',
    # 'tree_method': 'gpu_hist',
    # 'num_class': 27,
    # 'seed': 0,
    # 'max_depth': 2,
    # 'colsample_bytree': 0.36524046160303747,
    # 'colsample_bylevel': 0.7008644188368828,
    # 'grow_policy': 'lossguide',
    # 'lambda': 1-08,
    # 'alpha': 0.1,
    # 'subsample': 0.9,
    # 'eta': 0.01,
    # 'eval_metric': 'merror'}

    estimator_type: Literal["XGBoostClassifier"] = "XGBoostClassifier"
    objective: str = "multi:softprob"
    eval_metric: str | Callable | None = None
    missing: float = np.nan
    gamma: Optional[float] = None
    reg_alpha: Optional[float] = None
    reg_lambda: Optional[float] = None
    max_depth: Optional[int] = None
    max_leaves: Optional[int] = None
    max_bin: Optional[int] = None
    grow_policy: Optional[str] = None
    learning_rate: Optional[float] = None
    n_estimators: Optional[int] = None
    verbosity: Optional[int] = None
    booster: Optional[str] = None
    tree_method: Optional[str] = None
    n_jobs: Optional[int] = None
    min_child_weight: Optional[float] = None
    max_delta_step: Optional[float] = None
    subsample: Optional[float] = None
    sampling_method: Optional[str] = None
    colsample_bytree: Optional[float] = None
    colsample_bylevel: Optional[float] = None
    colsample_bynode: Optional[float] = None
    scale_pos_weight: Optional[float] = None
    random_state: Optional[int] = None


class XGBoostRegressor(pydantic.BaseModel):
    """https://xgboost.readthedocs.io/en/stable/parameter.html"""

    estimator_type: Literal["XGBoostRegressor"] = "XGBoostRegressor"
    objective: str = "reg:squarederror"
    eval_metric: str | Callable | None = None
    missing: float = np.nan
    gamma: Optional[float] = None
    reg_alpha: Optional[float] = None
    reg_lambda: Optional[float] = None
    max_depth: Optional[int] = None
    max_leaves: Optional[int] = None
    max_bin: Optional[int] = None
    grow_policy: Optional[str] = None
    learning_rate: Optional[float] = None
    n_estimators: Optional[int] = None
    verbosity: Optional[int] = None
    booster: Optional[str] = None
    tree_method: Optional[str] = None
    n_jobs: Optional[int] = None
    min_child_weight: Optional[float] = None
    max_delta_step: Optional[float] = None
    subsample: Optional[float] = None
    sampling_method: Optional[str] = None
    colsample_bytree: Optional[float] = None
    colsample_bylevel: Optional[float] = None
    colsample_bynode: Optional[float] = None
    scale_pos_weight: Optional[float] = None
    random_state: Optional[int] = None


class LinearRegression(pydantic.BaseModel):
    estimator_type: Literal["LinearRegression"] = "LinearRegression"


EstimatorType: TypeAlias = XGBoostClassifier | XGBoostRegressor | LinearRegression

EstimatorInterface = Annotated[
    EstimatorType, pydantic.Field(..., discriminator="estimator_type")
]
