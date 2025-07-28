import abc

from wiz.shared.estimator.estimator import FeatureArray  # type: ignore[import-untyped]
from wiz.shared.estimator.estimator import DoubleArray  # type: ignore[import-untyped]


class PostProc(abc.ABC):

    @abc.abstractmethod
    def transform(self, predictions: DoubleArray) -> DoubleArray: ...


class FittablePostProc(PostProc):

    @abc.abstractmethod
    def fit(self, features: FeatureArray, targets: DoubleArray) -> None: ...
