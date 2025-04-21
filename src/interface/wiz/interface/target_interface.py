from collections.abc import Sequence
import enum
from typing import Annotated, Literal, TypeAlias

import pydantic


class DummyTransformer(pydantic.BaseModel):
    target_type: Literal["DummyTransformer"] = "DummyTransformer"


class LogTransformer(pydantic.BaseModel):
    target_type: Literal["LogTransformer"] = "LogTransformer"


class PowerTransformer(pydantic.BaseModel):
    target_type: Literal["PowerTransformer"] = "PowerTransformer"
    l: float


TargetType: TypeAlias = DummyTransformer | LogTransformer | PowerTransformer

TargetInterface = Annotated[
    TargetType, pydantic.Field(..., discriminator="target_type")
]
