from typing import TypeAlias
from collections.abc import Mapping, Sequence
from sklearn import metrics
from sklearn import model_selection
from sklearn import linear_model
import polars as pl
import numpy as np
import pydantic
from wiz.post_proc.post_proc import FittablePostProc, DoubleArray, FeatureArray


class PrecisionRecallPoint(pydantic.BaseModel):
    precision: float
    recall: float
    threshold: float


PrecisionRecallCurveTransformer: TypeAlias = Mapping[int, PrecisionRecallPoint]


class PrecisionRecallCurve(pydantic.BaseModel):
    precisions: Sequence[int]
    points: Sequence[PrecisionRecallPoint]

    def to_transformer(self) -> PrecisionRecallCurveTransformer:
        return {
            precision: point for precision, point in zip(self.precisions, self.points)
        }


class PrecisionCalibrator(FittablePostProc):
    def __init__(self, *, precision_recall_curve: PrecisionRecallCurve | None = None):
        super().__init__()
        self.precision_recall_curve = precision_recall_curve
        self.precision_recall_curve_transformer = (
            precision_recall_curve.to_transformer() if precision_recall_curve else None
        )

    def fit(self, features: FeatureArray, targets: DoubleArray) -> None:
        pass

    def transform(self, predictions: DoubleArray) -> DoubleArray:
        return np.array(
            [
                self.precision_recall_curve_transformer[precision].threshold
                for precision in predictions
            ]
        )

    def transform_sparse(self, predictions: DoubleArray) -> DoubleArray:
        point_thresholds = np.array(
            [point.threshold for point in self.precision_recall_curve.points]
        )
        return (
            np.searchsorted(
                point_thresholds,
                predictions,
                side="right",
            )
            - 1
        )


if __name__ == "__main__":
    from sklearn import datasets  # type: ignore[import-untyped]
    from sklearn import decomposition  # type: ignore[import-untyped]
    import matplotlib.pyplot as plt

    PRINT_PCA: bool = False

    X, y = datasets.make_classification(
        n_samples=1_000_000, n_features=10, n_classes=2, random_state=42, class_sep=0.2
    )

    if PRINT_PCA:

        print(X.shape)
        print(y.shape)

        print(X)
        print(y)

        # Initialize PCA
        pca = decomposition.PCA()

        # Fit and transform the data
        X_pca = pca.fit_transform(X)

        # Print explained variance ratio
        print("\nExplained variance ratio:", pca.explained_variance_ratio_)
        print(
            "Cumulative explained variance ratio:",
            pca.explained_variance_ratio_.cumsum(),
        )

        # Print shape of transformed data
        print("\nShape after PCA:", X_pca.shape)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=33325
    )

    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    precision, recall, thresholds = metrics.precision_recall_curve(
        y_test, y_pred_proba[:, 1], pos_label=1
    )
    thresholds = np.concatenate(([-np.inf], thresholds))

    precision_recall_curve = [
        PrecisionRecallPoint(precision=p, recall=r, threshold=t)
        for p, r, t in zip(precision, recall, thresholds)
        if t >= 0
    ]
    precision_recall_curve_df = (
        pl.DataFrame(precision_recall_curve)
        .with_columns(
            (pl.col("precision") * 100)
            .floor()
            .cast(pl.Int8)
            .alias("precision_rounded_int")
        )
        .group_by("precision_rounded_int")
        .agg(
            pl.col("threshold").min().alias("threshold"),
            pl.col("precision").min().alias("precision"),
            pl.col("recall").min().alias("recall"),
        )
        .sort(by="threshold")
        .select(
            [
                pl.col("precision_rounded_int").alias("precision"),
                pl.struct(["precision", "recall", "threshold"])
                .map_elements(
                    PrecisionRecallPoint.model_validate,
                    return_dtype=pl.Object,
                )
                .alias("point"),
            ]
        )
    )

    print(
        PrecisionRecallCurve.model_validate(
            precision_recall_curve_df.to_dict(as_series=False)
        )
    )
    print(precision_recall_curve[-10:])
    print(precision_recall_curve_df)
    print(precision_recall_curve_df.shape)
    print(precision_recall_curve_df.head())
