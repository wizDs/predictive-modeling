from collections.abc import Sequence
from typing import final
import abc
import polars as pl
import pydantic
from wiz.interface import preproc_interface
from sklearn import compose, preprocessing, pipeline, impute

OUTPUT_TYPE = preproc_interface.OutputType.POLARS


class PreProcessor(abc.ABC):

    # @property
    # @abc.abstractmethod
    # def basic_columns(self) -> preproc_interface.BasicColumns: ...

    @abc.abstractmethod
    def fit(
        self, features: pl.DataFrame, targets: pl.DataFrame | None = None
    ) -> None: ...

    @abc.abstractmethod
    def _transform(self, features: pl.DataFrame, /) -> pl.DataFrame: ...

    @final
    def transform(self, features: pl.DataFrame, /) -> pl.DataFrame:
        return self._transform(features)


class DefaultPreProcessor(PreProcessor):

    def __init__(self, interface: preproc_interface.DefaultPreProcessor, /):
        super().__init__()
        self.pipeline = self.construct_pipeline(
            numerical_columns=interface.basic_columns.numerical_columns,
            categorical_columns=interface.basic_columns.categorical_columns,
            categorical_proc_type=interface.categorical_processor_type,
        )

    def fit(self, features: pl.DataFrame, targets: pl.DataFrame | None = None) -> None:
        if targets is not None:
            self.pipeline.fit(features, targets)
        else:
            self.pipeline.fit(features)

    def _transform(self, features: pl.DataFrame, /) -> pl.DataFrame:
        return self.pipeline.transform(features)

    def construct_pipeline(
        self,
        numerical_columns: Sequence[str],
        categorical_columns: Sequence[str],
        categorical_proc_type: preproc_interface.CategoricalProcessor,
    ) -> compose.ColumnTransformer:
        return compose.ColumnTransformer(
            transformers=[
                ("num", self._numerical_pipeline(), numerical_columns),
                (
                    "cat",
                    self._categorical_pipeline(categorical_proc_type),
                    categorical_columns,
                ),
            ],
            verbose_feature_names_out=False,
        ).set_output(transform=OUTPUT_TYPE)

    @staticmethod
    def _categorical_pipeline(
        proc_type: preproc_interface.CategoricalProcessor,
    ) -> pipeline.Pipeline:
        match proc_type:
            case preproc_interface.CategoricalProcessor.ORDINAL_ENCODER:
                preproc_step = preprocessing.OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                )
            case preproc_interface.CategoricalProcessor.ONE_HOT_ENCODER:
                preproc_step = preprocessing.OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    drop="first",
                    sparse_output=False,
                )
            case preproc_interface.CategoricalProcessor.TARGET_ENCODER:
                preproc_step = preprocessing.TargetEncoder()

        return pipeline.Pipeline(
            steps=[
                (
                    proc_type,
                    preproc_step.set_output(
                        transform=preproc_interface.OutputType.POLARS
                    ),
                ),
                (
                    "standard_scalar",
                    preprocessing.StandardScaler().set_output(transform=OUTPUT_TYPE),
                ),
            ]
        )

    @staticmethod
    def _numerical_pipeline(impute_value: float = -1.0) -> pipeline.Pipeline:
        return pipeline.Pipeline(
            steps=[
                (
                    "imputer",
                    impute.SimpleImputer(fill_value=impute_value).set_output(
                        transform=OUTPUT_TYPE
                    ),
                ),
                (
                    "scaler",
                    preprocessing.StandardScaler().set_output(transform=OUTPUT_TYPE),
                ),
            ]
        )
