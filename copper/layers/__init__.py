from .layer_base import ComputeLayer, TensorLayer

from .layer_base import TensorRef as TensorRef
from .layer_base import Program as Program
from .layer_base import Kernel as Kernel
from .layer_base import ShaderCursor as ShaderCursor
from .layer_base import ReflectedType as ReflectedType

from typing import Optional

# import importlib
# if importlib.util.find_spec("torch"):
#    from .torch_layer import TorchLayer
#
# if importlib.util.find_spec("falcor"):
#    from .falcor_layer import FalcorLayer

_tensor_layer: Optional[TensorLayer] = None
_compute_layer: Optional[ComputeLayer] = None


def tensor_layer() -> TensorLayer:
    assert _tensor_layer is not None
    return _tensor_layer


def compute_layer() -> ComputeLayer:
    assert _compute_layer is not None
    return _compute_layer


def tensor_ref() -> TensorRef:
    return tensor_layer().empty_ref()


def set_tensor_layer(l: TensorLayer):
    global _tensor_layer
    _tensor_layer = l


def set_compute_layer(l: ComputeLayer):
    global _compute_layer
    _compute_layer = l
