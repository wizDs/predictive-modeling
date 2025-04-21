import abc
import enum
from typing import Generic, ParamSpec, TypeVar, final
from wiz.interface import target_interface
import numpy as np


Target = TypeVar("Target")  # Return type
Params = ParamSpec("Params")  # Parameter specification for *args and **kwargs

EPSILON: float = 0.00001
LARGE_EXP: float = 30.0
LARGE: float = 99999999.0


class TargetTransformer(abc.ABC, Generic[Target]):

    @abc.abstractmethod
    def func(self, target: Target) -> Target: ...

    @abc.abstractmethod
    def inv_func(self, target: Target) -> Target: ...

    @final
    def transform(self, target: Target, inverse: bool = False) -> Target:
        if inverse:
            return self.inv_func(target)
        return self.func(target)


class LogTransformer(TargetTransformer[np.ndarray]):

    def func(self, target: np.ndarray) -> np.ndarray:
        return np.log(np.clip(target, EPSILON, LARGE))

    def inv_func(self, target: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(target, -100, LARGE_EXP))


class PowerTransformer(TargetTransformer[np.ndarray]):

    def __init__(self, interface: target_interface.PowerTransformer):
        super().__init__()
        self.l = interface.l

    def func(self, target: np.ndarray) -> np.ndarray:
        if self.l > 3:
            target = np.clip(target, 0, LARGE_EXP)
        return np.power(target, self.l)

    def inv_func(self, target: np.ndarray) -> np.ndarray:
        if 1 / self.l > 3:
            target = np.clip(target, 0, LARGE_EXP)
        return np.power(target, 1 / self.l)


class DummyTransformer(TargetTransformer[np.ndarray]):

    def func(self, target: np.ndarray) -> np.ndarray:
        return target

    def inv_func(self, target: np.ndarray) -> np.ndarray:
        return target
