

from typing import Any, Optional, Sequence

from sgl import Device
from .basetype import BaseType
from .basevariable import BaseVariable
from .enums import AccessType


class BaseTypeImpl(BaseType):
    def __init__(self):
        super().__init__()

    def has_derivative(self, value: Any = None) -> bool:
        return False

    def is_writable(self, value: Any = None) -> bool:
        return False

    def differentiable(self, value: Any = None):
        return False

    def differentiate(self, value: Any = None):
        return None

    def container_shape(self, value: Any = None) -> Sequence[Optional[int]]:
        return ()

    def shape(self, value: Any = None):
        return tuple(self.container_shape(value)) + tuple(self.element_type(value).shape())

    def create_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: Any) -> Any:
        pass

    def read_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: Any, result: Any) -> None:
        pass
