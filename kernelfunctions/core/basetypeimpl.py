

from typing import TYPE_CHECKING, Any

from kernelfunctions.backend import Device
from kernelfunctions.shapes import TLooseShape

from .basetype import BaseType

if TYPE_CHECKING:
    from .boundvariableruntime import BoundVariableRuntime


class BaseTypeImpl(BaseType):
    def __init__(self):
        super().__init__()

    @property
    def has_derivative(self) -> bool:
        return False

    @property
    def is_writable(self) -> bool:
        return False

    @property
    def differentiable(self):
        return False

    @property
    def derivative(self):
        return None

    def get_container_shape(self, value: Any = None) -> TLooseShape:
        return ()

    def get_shape(self, value: Any = None) -> TLooseShape:
        return self.get_container_shape(value) + self.element_type.get_shape()

    def create_calldata(self, device: Device, binding: 'BoundVariableRuntime', broadcast: list[bool], data: Any) -> Any:
        pass

    def read_calldata(self, device: Device, binding: 'BoundVariableRuntime', data: Any, result: Any) -> None:
        pass

    def update_from_bound_type(self, bound_type: 'BaseType'):
        pass
