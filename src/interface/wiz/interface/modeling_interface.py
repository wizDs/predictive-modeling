from wiz.interface import estimator_interface, preproc_interface, target_interface
import dataclasses


@dataclasses.dataclass
class TrainInputInterface:
    preprocessor: preproc_interface.PreProcInterface
    estimator: estimator_interface.EstimatorInterface
    target: target_interface.TargetInterface
