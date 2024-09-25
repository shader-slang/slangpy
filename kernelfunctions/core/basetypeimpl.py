

from typing import TYPE_CHECKING, Any, Optional, Sequence

from kernelfunctions.backend import Device

from .basetype import BaseType

if TYPE_CHECKING:
    from .basevariable import BoundVariable


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

    def container_shape(self, value: Any = None) -> Sequence[Optional[int]]:
        return ()

    def shape(self, value: Any = None):
        return tuple(self.container_shape(value)) + tuple(self.element_type.shape())

    def create_calldata(self, device: Device, input_value: 'BoundVariable', broadcast: list[bool], data: Any) -> Any:
        pass

    def read_calldata(self, device: Device, input_value: 'BoundVariable', data: Any, result: Any) -> None:
        pass

    def update_from_bound_type(self, bound_type: 'BaseType'):
        pass
