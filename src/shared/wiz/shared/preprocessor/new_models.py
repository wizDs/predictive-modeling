from collections.abc import Sequence
from dataclasses import dataclass
import abc
import enum
import pathlib
import math
from functools import partial
from typing import Callable, Generic, Iterable, ParamSpec, Self, TypeVar, final
import json
import numpy as np
import pydantic
from sklearn import pipeline
from wiz.interface import estimator_interface, preproc_interface
from wiz.shared import get_model
import xgboost
from toolz.itertoolz import pluck
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from wiz.shared.estimator import estimator
from wiz.shared.preprocessor import preprocessor
from wiz.shared.target_transformer import target_transformer

# from .eval_regression import ModelReport, ModelReportBuilder

# a frunction that returns an esimator, that is a full ds-model
ModelConstructor = Callable[[None], BaseEstimator]
PreprocessorConstructor = Callable[[None], TransformerMixin]


def create_xgboost(preprocessor: PreprocessorConstructor = None) -> Pipeline:
    """gets a new instance of a xgboost model"""
    if not preprocessor:
        preprocessor = create_preprocessor_ohe
    return Pipeline(
        steps=[("preprocessor", preprocessor()), ("model", xgboost.XGBRegressor())]
    )


def create_log_linear(preprocessor: PreprocessorConstructor = None) -> Pipeline:
    @np.vectorize
    def truncated_exp(x: float) -> float:
        MIN_PRICE = 0
        MAX_PRICE = 600_000
        return max(min(math.exp(x), MAX_PRICE), MIN_PRICE)

    if not preprocessor:
        preprocessor = create_preprocessor_ohe

    tt_linear = TransformedTargetRegressor(
        regressor=LinearRegression(n_jobs=-1),
        func=np.log,
        inverse_func=truncated_exp,
        check_inverse=False,
    )

    return Pipeline(steps=[("preprocessor", preprocessor()), ("model", tt_linear)])


def create_knn(preprocessor: PreprocessorConstructor = None, **kwargs) -> Pipeline:
    if not preprocessor:
        preprocessor = create_preprocessor_ohe

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor()),
            ("model", KNeighborsRegressor(**kwargs)),
        ]
    )


def create_lasso(preprocessor: PreprocessorConstructor = None) -> Pipeline:
    if not preprocessor:
        preprocessor = create_preprocessor_ohe

    return Pipeline(
        steps=[("preprocessor", preprocessor()), ("model", Lasso(tol=1e-3))]
    )


# def error_by_actual_price(
#     features: pl.DataFrame,
#     targets: pl.DataFrame,
#     model: BaseEstimator,
#     kfold: KFold,
#     error_measure: Callable = None,
# ) -> pd.DataFrame:
#     """get table with prediction error by actual price"""

#     def mean_absolute_error(y_test, y_pred):
#         return np.abs(y_test - y_pred)

#     if not error_measure:
#         error_measure = mean_absolute_error
#     train_index, test_index = next(kfold.split(features))
#     train_index, test_index = pl.Series(train_index), pl.Series(test_index)

#     # evaluate model
#     X_train, X_test = features.filter(train_index), features.filter(test_index)
#     y_train, y_test = targets.filter(train_index), targets.filter(test_index)

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     eval_df = pd.DataFrame(
#         {
#             "price": y_test,
#             "pred_price": y_pred.round(),
#             "error_pred": error_measure(y_test, y_pred).round(),
#         }
#     )

#     return (
#         eval_df.groupby(pd.cut(x=eval_df["price"] / 1_000, bins=range(0, 650, 20)))
#         .agg(
#             error_pred=("error_pred", "mean"),
#             std_pred=("error_pred", "std"),
#             count=("error_pred", "count"),
#         )
#         .round(0)
#         .astype("Int64")
#         .loc[lambda x: x["count"] > 0]
#     )


# @dataclass
# class EvaluationSet:
#     features: pd.DataFrame
#     labels: pd.Series
#     model_constructor: ModelConstructor


# def evaluate_multiple_models(k_fold: KFold, /, *args) -> Iterable[ModelReport]:
#     for eval_set in args:
#         report = ModelReportBuilder(
#             eval_set.features, eval_set.labels, eval_set.model_constructor(), k_fold
#         )
#         yield report


# read data
def read_data(path: pathlib.Path):
    return (
        pl.read_csv(path, null_values="NA")
        .rename(
            mapping=lambda s: s.lower()
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )
        .with_columns(pl.col(pl.String).replace("None", None))
    )


if __name__ == "__main__":

    # num_cols = make_column_selector(dtype_include=np.number)
    # cat_cols = make_column_selector(dtype_include=object)

    curr_path = pathlib.Path(".").absolute()
    data_path = curr_path / "data" / "train.csv"
    print(data_path)

    # read feature descriptions
    with open(
        curr_path / "data" / "feature_description.txt", "r", encoding="utf-8"
    ) as f:
        feature_descriptions = json.loads(f.read())["features"]

    # mapper from name to description
    description_mapper = dict(pluck(["name", "desc"], feature_descriptions))

    data_df = read_data(data_path)

    test_data_df = data_df.sample(n=len(data_df) * 0.2)
    train_data_df = data_df.join(test_data_df, on="id", how="anti")

    assert len(set(data_df["id"])) == len(data_df["id"]), "id must be unique"
    assert not set(data_df["id"]) - (
        set(test_data_df["id"]) | set(train_data_df["id"])
    ), "train-test-split must cover all ids"

    test_features_df = test_data_df.drop("id", "saleprice")
    test_targets_df = test_data_df.select("saleprice")

    features_df = train_data_df.drop("id", "saleprice")
    targets_df = train_data_df.select("saleprice")

    class TrainInputInterface(pydantic.BaseModel):
        preprocessor: preproc_interface.DefaultPreProcessor
        estimator: estimator_interface.EstimatorInterface
        # target_transformer: target_transformer.TargetTransformer | None = None

    # for col in features_df[cat_cols].columns:
    #     features_df[col] = features_df[col].fillna("")
    # X[num_cols].pipe(num_ppl.fit_transform).describe().transpose().round(2)

    num_cols = features_df.select(pl.selectors.numeric()).columns
    cat_cols = features_df.select(pl.selectors.string()).columns
    basic_columns = preproc_interface.BasicColumns(
        numerical_columns=num_cols, categorical_columns=cat_cols
    )
    num_columns_only = preproc_interface.BasicColumns(
        numerical_columns=num_cols, categorical_columns=[]
    )

    pre_proc_target_encoder = preprocessor.DefaultPreProcessor(
        basic_columns=basic_columns,
        categorical_proc_type=preproc_interface.CategoricalProcessor.TARGET_ENCODER,
    )
    pre_proc_ohe = preprocessor.DefaultPreProcessor(
        basic_columns=basic_columns,
        categorical_proc_type=preproc_interface.CategoricalProcessor.ONE_HOT_ENCODER,
    )
    pre_proc_oe = preprocessor.DefaultPreProcessor(
        basic_columns=basic_columns,
        categorical_proc_type=preproc_interface.CategoricalProcessor.ORDINAL_ENCODER,
    )
    pre_proc_num_only = preprocessor.DefaultPreProcessor(
        basic_columns=num_columns_only,
        categorical_proc_type=preproc_interface.CategoricalProcessor.ONE_HOT_ENCODER,
    )

    pre_proc_ohe.fit(features_df)
    pre_proc_oe.fit(features_df)
    pre_proc_num_only.fit(features_df)
    pre_proc_target_encoder.fit(features_df, targets_df)

    log_transformer = target_transformer.LnTransformer()
    _log_targets_df = log_transformer.transform(targets_df["saleprice"].to_numpy())

    # normal
    model = get_model.model_from_type(estimator_interface.XGBoostRegressor())
    model.fit(
        features=pre_proc_ohe.transform(features_df).to_numpy(),
        targets=targets_df["saleprice"].to_numpy(),
    )

    _pred = model.predict(features=pre_proc_ohe.transform(test_features_df))
    _target = test_targets_df["saleprice"].to_numpy()
    metrics.mean_absolute_percentage_error(_target, _pred)

    # # log
    # model = XGBoostRegressor(estimator=modeling_interface.XGBoostRegressor())
    # model.fit(
    #     features=pre_proc.transform(features_df).to_numpy(), targets=_log_targets_df
    # )

    # _log_pred = model.predict(features=pre_proc.transform(test_features_df))
    # _pred = log_transformer.transform(_log_pred, inverse=True).round()
    # _target = test_targets_df["saleprice"].to_numpy()

    # metrics.mean_absolute_percentage_error(_target, _pred)

    # pre_proc_num_only
    # normal
    model = get_model.model_from_type(estimator_interface.XGBoostRegressor())
    model.fit(
        features=pre_proc_num_only.transform(features_df).to_numpy(),
        targets=targets_df["saleprice"].to_numpy(),
    )

    _pred = model.predict(features=pre_proc_num_only.transform(test_features_df))
    _target = test_targets_df["saleprice"].to_numpy()
    print(
        "normal-num_only",
        round(metrics.mean_absolute_percentage_error(_target, _pred), 3),
        round(metrics.median_absolute_error(_target, _pred), 3),
        round(metrics.mean_absolute_error(_target, _pred), 3),
    )

    # pre_proc_oe
    # normal
    model = get_model.model_from_type(estimator_interface.XGBoostRegressor())
    model.fit(
        features=pre_proc_oe.transform(features_df).to_numpy(),
        targets=targets_df["saleprice"].to_numpy(),
    )

    _pred = model.predict(features=pre_proc_oe.transform(test_features_df))
    _target = test_targets_df["saleprice"].to_numpy()
    print(
        "normal-oe",
        round(metrics.mean_absolute_percentage_error(_target, _pred), 3),
        round(metrics.median_absolute_error(_target, _pred), 3),
        round(metrics.mean_absolute_error(_target, _pred), 3),
    )

    # pre_proc_ohe
    # normal
    model = get_model.model_from_type(estimator_interface.XGBoostRegressor())
    model.fit(
        features=pre_proc_ohe.transform(features_df).to_numpy(),
        targets=targets_df["saleprice"].to_numpy(),
    )

    _pred = model.predict(features=pre_proc_ohe.transform(test_features_df))
    _target = test_targets_df["saleprice"].to_numpy()
    print(
        "normal-ohe",
        round(metrics.mean_absolute_percentage_error(_target, _pred), 3),
        round(metrics.median_absolute_error(_target, _pred), 3),
        round(metrics.mean_absolute_error(_target, _pred), 3),
    )

    # pre_proc_target_encoder
    # normal
    model = get_model.model_from_type(estimator_interface.XGBoostRegressor())
    model.fit(
        features=pre_proc_target_encoder.transform(features_df).to_numpy(),
        targets=targets_df["saleprice"].to_numpy(),
    )

    _pred = model.predict(features=pre_proc_target_encoder.transform(test_features_df))
    _target = test_targets_df["saleprice"].to_numpy()
    print(
        "normal-target-encoder",
        round(metrics.mean_absolute_percentage_error(_target, _pred), 3),
        round(metrics.median_absolute_error(_target, _pred), 3),
        round(metrics.mean_absolute_error(_target, _pred), 3),
    )

    # where
    test_features_df.with_columns(
        [
            pl.Series(name="prediction", values=_pred, dtype=pl.Float64),
            pl.Series(name="target", values=_target, dtype=pl.Float64),
        ]
    ).with_columns(
        (pl.col("target") / 1_000)
        .qcut(quantiles=np.arange(0, 1.01, 0.1), labels=np.arange(0, 1.01, 0.1))
        .alias("category")
    )

    # kfold = KFold(random_state=423, shuffle=True)
    # ksplit = kfold.split(features_df)
    # next(ksplit)

    distr_df = error_by_actual_price(features_df, targets_df, create_xgboost(), kfold)
    print(distr_df)

    evaluation_set = [
        EvaluationSet(
            features=features_df[num_cols],
            labels=targets_df,
            model_constructor=create_xgboost,
        ),
        EvaluationSet(
            features=features_df, labels=targets_df, model_constructor=create_xgboost
        ),
        EvaluationSet(
            features=features_df,
            labels=targets_df,
            model_constructor=partial(
                create_xgboost, preprocessor=create_preprocessor_oe
            ),
        ),
        EvaluationSet(
            features=features_df[num_cols],
            labels=targets_df,
            model_constructor=partial(
                create_log_linear, preprocessor=create_preprocessor_oe
            ),
        ),
        EvaluationSet(
            features=features_df[
                features_df[num_cols].columns.tolist()
                + ["poolqc", "fence", "paveddrive"]
            ],
            labels=targets_df,
            model_constructor=partial(
                create_log_linear, preprocessor=create_preprocessor_oe
            ),
        ),
        EvaluationSet(
            features=features_df[
                features_df[num_cols].columns.tolist()
                + ["poolqc", "fence", "paveddrive"]
            ],
            labels=targets_df,
            model_constructor=create_log_linear,
        ),
        EvaluationSet(
            features=features_df[num_cols],
            labels=targets_df,
            model_constructor=create_lasso,
        ),
        EvaluationSet(
            features=features_df, labels=targets_df, model_constructor=create_knn
        ),
        EvaluationSet(
            features=features_df,
            labels=targets_df,
            model_constructor=partial(create_knn, preprocessor=create_preprocessor_oe),
        ),
    ]

    reports = evaluate_multiple_models(kfold, *evaluation_set)
    for eval_set, r in zip(evaluation_set, reports):
        print("\n\n")
        print(
            eval_set.model_constructor
            if isinstance(eval_set.model_constructor, partial)
            else eval_set.model_constructor.__name__
        )
        print("column counts: ", len(eval_set.features.columns.tolist()))
        print(json.dumps(r.summary.model_dump(), indent=4))

    # test data
    test_path = curr_path / "data" / "test.csv"
    test_df = pd.read_csv(test_path)
    test_df.set_index("Id", inplace=True)
    X_validation = test_df.rename(
        columns=lambda s: (
            s.lower().replace(".", "_").replace("(", "").replace(")", "")
        )
    )

    model = create_xgboost()
    model.fit(features_df, targets_df)
    y_validation_pred = model.predict(X_validation).round().astype(int)
    y_validation_pred
