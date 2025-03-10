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
from wiz.interface import estimator_interface, preproc_interface, target_interface
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


@dataclass
class EvaluationSet:
    train_features: pl.DataFrame
    train_targets: np.ndarray
    test_features: pl.DataFrame
    test_targets: np.ndarray


def split_train_test(df: pl.DataFrame, /):

    test_data_df = df.sample(n=len(df) * 0.2)
    train_data_df = df.join(test_data_df, on="id", how="anti")

    assert len(set(df["id"])) == len(df["id"]), "id must be unique"
    assert not set(df["id"]) - (
        set(test_data_df["id"]) | set(train_data_df["id"])
    ), "train-test-split must cover all ids"

    return EvaluationSet(
        train_features=train_data_df.drop("id", "saleprice"),
        train_targets=train_data_df.get_column("saleprice").to_numpy(),
        test_features=test_data_df.drop("id", "saleprice"),
        test_targets=test_data_df.get_column("saleprice").to_numpy(),
    )


@dataclass
class TrainInputInterface:
    preprocessor: preproc_interface.PreProcInterface
    target: target_interface.TargetInterface
    estimator: estimator_interface.EstimatorInterface
    # target_transformer: target_transformer.TargetTransformer | None = None


def evaluate_estimator(
    train_interface: TrainInputInterface,
    data: EvaluationSet,
) -> dict[str, float]:
    preproc = get_model.preprocessor_from_type(train_interface.preprocessor)
    _estimator = get_model.model_from_type(train_interface.estimator)
    _tt = get_model.target_from_type(train_interface.target)
    preproc.fit(data.train_features, _tt.func(data.train_targets))

    _estimator.fit(
        preproc.transform(data.train_features).to_numpy(), _tt.func(data.train_targets)
    )
    tt_predictions = _estimator.predict(
        preproc.transform(data.test_features).to_numpy()
    )
    predictions = _tt.inv_func(tt_predictions)

    return {
        "mape": round(
            metrics.mean_absolute_percentage_error(data.test_targets, predictions),
            3,
        ),
        "mae": round(metrics.mean_absolute_error(data.test_targets, predictions)),
        "rmse": round(
            metrics.root_mean_squared_error(data.test_targets, predictions), 3
        ),
        "mean_error": round(np.mean(predictions - data.test_targets)),
        "median_error": round(np.median(predictions - data.test_targets)),
    }


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

    # for col in features_df[cat_cols].columns:
    #     features_df[col] = features_df[col].fillna("")
    # X[num_cols].pipe(num_ppl.fit_transform).describe().transpose().round(2)
    _data_df = data_df.drop("id", "saleprice")
    basic_columns = preproc_interface.BasicColumns(
        numerical_columns=_data_df.select(pl.selectors.numeric()).columns,
        categorical_columns=_data_df.select(pl.selectors.string()).columns,
    )
    num_columns_only = preproc_interface.BasicColumns(
        numerical_columns=_data_df.select(pl.selectors.numeric()).columns,
        categorical_columns=[],
    )

    output = []
    for i in range(30):
        train_validation_df = split_train_test(data_df)

        runs = [
            # (
            #     "num_columns_only",
            #     num_columns_only,
            #     preproc_interface.CategoricalProcessor.ONE_HOT_ENCODER,
            # ),
            # (
            #     "basic_columns",
            #     basic_columns,
            #     preproc_interface.CategoricalProcessor.ONE_HOT_ENCODER,
            # ),
            (
                "basic_columns",
                basic_columns,
                preproc_interface.CategoricalProcessor.ORDINAL_ENCODER,
            ),
            (
                "basic_columns",
                basic_columns,
                preproc_interface.CategoricalProcessor.TARGET_ENCODER,
            ),
        ]

        for name, _basic_columns, proc_type in runs:
            for tt in (
                target_interface.DummyTransformer(),
                target_interface.PowerTransformer(l=1.5),
                target_interface.PowerTransformer(l=0.5),
                target_interface.LogTransformer(),
            ):
                for est in (
                    estimator_interface.XGBoostRegressor(),
                    # estimator_interface.LinearRegression(),
                    # estimator_interface.LGBMRegressor(),
                ):
                    train_interface = TrainInputInterface(
                        preprocessor=preproc_interface.PreProcInterface(
                            basic_columns=_basic_columns,
                            categorical_processor_type=proc_type,
                        ),
                        target=tt,
                        estimator=est,
                    )
                    eval_metrics = evaluate_estimator(
                        train_interface=train_interface,
                        data=train_validation_df,
                    )

                    output += [
                        {
                            # "name": name,
                            "estimator": est.estimator_type,
                            "type": proc_type.value,
                            "target_transformer": (
                                tt.target_type
                                if not isinstance(tt, target_interface.PowerTransformer)
                                else f"{tt.target_type}({tt.l})"
                            ),
                            **eval_metrics,
                        }
                    ]

    _df = (
        pl.DataFrame(output)
        .group_by(["estimator", "type", "target_transformer"])
        .agg(
            [
                pl.mean("mape").round(3).alias("avg_mape"),
                pl.mean("mean_error").cast(pl.Int32).alias("mean_error"),
                pl.mean("median_error").cast(pl.Int32).alias("median_error"),
                pl.mean("mae").cast(pl.Int32).alias("avg_mae"),
                pl.std("mae").cast(pl.Int32).alias("std_mae"),
            ]
        )
        .sort("avg_mape")
    )
    print(_df.to_pandas())

    log_transformer = target_transformer.LogTransformer()
    _log_targets_df = log_transformer.transform(targets_df["saleprice"].to_numpy())

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
