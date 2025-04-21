from collections.abc import Sequence
import enum

import pydantic


class CategoricalProcessor(enum.StrEnum):
    ONE_HOT_ENCODER = "one_hot_encoder"
    ORDINAL_ENCODER = "ordinal_encoder"
    TARGET_ENCODER = "target_encoder"


class OutputType(enum.StrEnum):
    PANDAS = "pandas"
    POLARS = "polars"


class BasicColumns(pydantic.BaseModel):
    numerical_columns: Sequence[str]
    categorical_columns: Sequence[str]
    target_column: str


class PreProcInterface(pydantic.BaseModel):
    basic_columns: BasicColumns
    categorical_processor_type: CategoricalProcessor
