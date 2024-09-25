

from typing import TYPE_CHECKING, Any, Optional, Sequence

from kernelfunctions.backend import Device

from .basetype import BaseType
from .enums import AccessType

if TYPE_CHECKING:
    from .basevariable import BaseVariable


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

    def create_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], broadcast: list[bool], data: Any) -> Any:
        pass

    def read_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: Any, result: Any) -> None:
        pass

    def update_from_bound_type(self, bound_type: 'BaseType'):
        pass
