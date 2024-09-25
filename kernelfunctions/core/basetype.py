

from typing import Any, Optional, Sequence, TYPE_CHECKING

import numpy.typing as npt

from kernelfunctions.backend import Device

from .enums import AccessType
from .codegen import CodeGenBlock

if TYPE_CHECKING:
    from .basevariable import BoundVariable


class BaseType:
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def has_derivative(self, value: Any = None) -> bool:
        raise NotImplementedError()

    def is_writable(self, value: Any = None) -> bool:
        raise NotImplementedError()

    def differentiable(self, value: Any = None) -> bool:
        raise NotImplementedError()

    def differentiate(self, value: Any = None) -> Optional['BaseType']:
        raise NotImplementedError()

    def container_shape(self, value: Any = None) -> Sequence[Optional[int]]:
        raise NotImplementedError()

    def element_type(self, value: Any = None) -> 'BaseType':
        raise NotImplementedError()

    def byte_size(self, value: Any = None) -> int:
        raise NotImplementedError()

    def shape(self, value: Any = None) -> Sequence[Optional[int]]:
        raise NotImplementedError()

    def python_return_value_type(self, value: Any = None) -> type:
        raise NotImplementedError()

    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BoundVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        raise NotImplementedError()

    def gen_load_store(self, cgb: CodeGenBlock, input_value: 'BoundVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        raise NotImplementedError()

    def create_calldata(self, device: Device, input_value: 'BoundVariable', access: tuple[AccessType, AccessType], broadcast: list[bool], data: Any) -> Any:
        raise NotImplementedError()

    def read_calldata(self, device: Device, input_value: 'BoundVariable', access: tuple[AccessType, AccessType], data: Any, result: Any) -> None:
        raise NotImplementedError()

    def allocate_return_value(self, device: Device, input_value: 'BoundVariable', slang_value: 'BoundVariable', data: Any, access: tuple[AccessType, AccessType]):
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
