import abc
import numpy as np

DoubleArray = np.ndarray[tuple[np.float64], np.dtype[np.float64]]
FeatureArray = np.ndarray[tuple[np.float64], np.dtype[np.float64]]


class PostProc(abc.ABC):

    @abc.abstractmethod
    def transform(self, predictions: DoubleArray) -> DoubleArray: ...


class FittablePostProc(PostProc):

    @abc.abstractmethod
    def fit(self, features: FeatureArray, targets: DoubleArray) -> None: ...
