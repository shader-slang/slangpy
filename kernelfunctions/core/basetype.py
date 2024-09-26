

from typing import Any, Optional, Sequence, TYPE_CHECKING

import numpy.typing as npt

from kernelfunctions.backend import Device
from kernelfunctions.shapes import TLooseShape

from .enums import AccessType
from .codegen import CodeGenBlock

if TYPE_CHECKING:
    from .boundvariable import BoundVariable
    from .boundvariableruntime import BoundVariableRuntime


class BaseType:
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def has_derivative(self) -> bool:
        raise NotImplementedError()

    @property
    def is_writable(self) -> bool:
        raise NotImplementedError()

    @property
    def differentiable(self) -> bool:
        raise NotImplementedError()

    @property
    def derivative(self) -> Optional['BaseType']:
        raise NotImplementedError()

    @property
    def element_type(self) -> 'BaseType':
        raise NotImplementedError()

    @property
    def python_return_value_type(self) -> type:
        raise NotImplementedError()

    def get_container_shape(self, value: Any = None) -> TLooseShape:
        raise NotImplementedError()

    def get_shape(self, value: Any = None) -> TLooseShape:
        raise NotImplementedError()

    def get_byte_size(self, value: Any = None) -> int:
        raise NotImplementedError()

    def gen_calldata(self, cgb: CodeGenBlock, binding: 'BoundVariable'):
        raise NotImplementedError()

    def create_calldata(self, device: Device, binding: 'BoundVariableRuntime', broadcast: list[bool], data: Any) -> Any:
        raise NotImplementedError()

    def read_calldata(self, device: Device, binding: 'BoundVariableRuntime', data: Any, result: Any) -> None:
        raise NotImplementedError()

    def allocate_return_value(self, device: Device, binding: 'BoundVariable', slang_value: 'BoundVariable', data: Any, access: tuple[AccessType, AccessType]):
        raise NotImplementedError()

    def create_output(self, device: Device, call_shape: Sequence[int]) -> Any:
        raise NotImplementedError()

    def read_output(self, device: Device, data: Any) -> Any:
        raise NotImplementedError()

    def to_numpy(self, value: Any) -> npt.NDArray[Any]:
        raise NotImplementedError()

    def from_numpy(self, array: npt.ArrayLike) -> Any:
        raise NotImplementedError()

    def update_from_bound_type(self, bound_type: 'BaseType'):
        raise NotImplementedError()
