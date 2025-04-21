import dataclasses
import polars as pl
import numpy as np
from sklearn import metrics
from wiz.shared import get_model
from wiz.interface import modeling_interface


@dataclasses.dataclass
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


def evaluate_estimator(
    interface: modeling_interface.TrainInputInterface,
    data: EvaluationSet,
) -> dict[str, float]:
    preproc = get_model.preprocessor_from_type(interface.preprocessor)
    _estimator = get_model.model_from_type(interface.estimator)
    _tt = get_model.target_from_type(interface.target)
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
