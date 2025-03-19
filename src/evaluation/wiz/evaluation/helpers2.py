from collections.abc import Sequence
import dataclasses
from typing import NamedTuple
import polars as pl
import numpy as np
import pydantic
from sklearn import metrics
from wiz.shared import get_model
from wiz.interface import modeling_interface


@dataclasses.dataclass
class EvaluationSet:
    train_dataset: pl.DataFrame
    infer_dataset: pl.DataFrame


class PredictionSet(pydantic.BaseModel):
    prediction: float
    target: float


def split_train_test(df: pl.DataFrame, /):

    test_data_df = df.sample(n=len(df) * 0.2)
    train_data_df = df.join(test_data_df, on="id", how="anti")

    assert len(set(df["id"])) == len(df["id"]), "id must be unique"
    assert not set(df["id"]) - (
        set(test_data_df["id"]) | set(train_data_df["id"])
    ), "train-test-split must cover all ids"

    return EvaluationSet(
        train_dataset=train_data_df.drop("id"),
        infer_dataset=test_data_df.drop("id"),
    )


def evaluate_estimator(
    interface: modeling_interface.TrainInputInterface,
    data: EvaluationSet,
) -> dict[str, float]:
    prediction_sets = get_predictions(interface, data)
    targets = np.array([p.target for p in prediction_sets])
    predictions = np.array([p.prediction for p in prediction_sets])
    return {
        "mape": round(
            metrics.mean_absolute_percentage_error(targets, predictions),
            3,
        ),
        "mae": round(metrics.mean_absolute_error(targets, predictions)),
        "rmse": round(metrics.root_mean_squared_error(targets, predictions), 3),
        "mean_error": round(np.mean(predictions - targets)),
        "median_error": round(np.median(predictions - targets)),
    }


def get_predictions(
    interface: modeling_interface.TrainInputInterface,
    data: EvaluationSet,
) -> Sequence[PredictionSet]:
    preproc = get_model.preprocessor_from_type(interface.preprocessor)
    _estimator = get_model.model_from_type(interface.estimator)
    _tt = get_model.target_from_type(interface.target)
    _target_column = interface.preprocessor.basic_columns.target_column
    _target_train = data.train_dataset[_target_column].to_numpy()
    _target_infer = data.infer_dataset[_target_column].to_numpy()
    preproc.fit(data.train_dataset, _target_train)

    _estimator.fit(
        preproc.transform(data.train_dataset).to_numpy(),
        _tt.func(_target_train),
    )
    tt_predictions = _estimator.predict(
        preproc.transform(data.infer_dataset).to_numpy()
    )
    predictions = _tt.inv_func(tt_predictions)
    return [
        PredictionSet(target=t, prediction=p)
        for t, p in zip(_target_infer, predictions)
    ]
