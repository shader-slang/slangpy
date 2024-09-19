from __future__ import annotations

import enum

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..reflection import ScalarKind, SlangType, SlangFunc
    from ..invokables import InvokableSlangFunc


class TensorLayer:
    def is_tensor(self, t: Any) -> bool:
        raise NotImplementedError

    def wrap_tensor(self, raw_tensor: Any) -> TensorRef:
        raise NotImplementedError

    def import_tensor(self, dlpack_ndarray: Any, buffer: BufferRef) -> TensorRef:
        raise NotImplementedError

    def meta_tensor(self, dtype: ScalarKind, shape: tuple[int, ...]) -> TensorRef:
        raise NotImplementedError

    def empty_ref(self) -> TensorRef:
        raise NotImplementedError

    def wrap_kernel_call(
        self, func: InvokableSlangFunc, root_var: Variable, values: list[Any]
    ):
        raise NotImplementedError

    def device(self) -> Device:
        raise NotImplementedError


class ComputeLayer:
    def extend_program(
        self,
        prog: Program,
        new_module_name: str,
        new_module_src: str,
        entry_point: Optional[str] = None,
    ) -> Program:
        raise NotImplementedError

    def create_kernel(self, prog: Program) -> Kernel:
        raise NotImplementedError

    def from_raw_program(self, raw_prog: Any) -> Program:
        raise NotImplementedError

    def device(self) -> Device:
        raise NotImplementedError

    def enable_printing(self, value: bool = True):
        raise NotImplementedError


class DeviceKind(enum.Enum):
    CPU = enum.auto()
    Cuda = enum.auto()
    Other = enum.auto()


class Device:
    def kind(self) -> DeviceKind:
        raise NotImplementedError

    def idx(self) -> int:
        raise NotImplementedError

    def sync(self):
        raise NotImplementedError

    def stream(self):
        raise NotImplementedError


class BufferRef:
    def __init__(self, tag: str = ""):
        super().__init__()
        self.tag = tag
        self.version = 0

    def data_ptr(self):
        raise NotImplementedError

    def device(self) -> Device:
        raise NotImplementedError


class Program:
    def find_type(self, name: str) -> Optional[ReflectedType]:
        raise NotImplementedError

    def find_function(self, name: str) -> Optional[SlangFunc]:
        raise NotImplementedError


class ReflectedType:
    def type(self) -> Optional[SlangType]:
        raise NotImplementedError

    def methods(self) -> list[SlangFunc]:
        raise NotImplementedError

    def generic_args(self) -> list[SlangType | int]:
        raise NotImplementedError


class Kernel:
    def rootvar(self) -> ShaderCursor:
        raise NotImplementedError

    def dispatch(self, *dims: int):
        raise NotImplementedError


class ShaderCursor:
    def set(self, value: Any):
        raise NotImplementedError

    def set_tensor(self, tensor: TensorRef, readable: bool, writeable: bool):
        raise NotImplementedError

    def __getitem__(self, idx: int | str) -> ShaderCursor:
        raise NotImplementedError

    def __setitem__(self, idx: int | str, value: Any):
        self[idx].set(value)


class TensorRef:
    def __init__(self):
        super().__init__()
        buf = self.buffer()
        self.version = 0 if buf is None else buf.version

    def is_empty(self) -> bool:
        raise NotImplementedError

    def get_dtype(self) -> SlangType:
        raise NotImplementedError

    def get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def get_ndim(self) -> int:
        return len(self.get_shape())

    def get_strides(self) -> tuple[int, ...]:
        raise NotImplementedError

    def get_offset(self) -> int:
        raise NotImplementedError

    def copy_data(self, src: TensorRef):
        raise NotImplementedError

    def create_view(
        self, offset: int, shape: tuple[int, ...], strides: tuple[int, ...]
    ) -> TensorRef:
        raise NotImplementedError

    def make_contiguous(self) -> TensorRef:
        raise NotImplementedError

    def point_to(self, other: TensorRef):
        raise NotImplementedError

    def unwrap(self) -> Any:
        raise NotImplementedError

    def buffer(self) -> Optional[BufferRef]:
        raise NotImplementedError

    def valid(self):
        buf = self.buffer()
        return buf is None or self.version == buf.version

    def check_valid(self):
        if not self.valid():
            buf = self.buffer()
            assert buf is not None
            msg = (
                f"Trying to access tensor owned by {buf.tag} "
                f"that was overwritten with different data in the meantime. This is likely because '{buf.tag}' "
                "was called multiple times. To avoid this error, call .clone() on this tensor before it is overwritten."
            )
            raise RuntimeError(msg)

    @staticmethod
    def broadcast(tensor: TensorRef, shape: tuple[int, ...]) -> TensorRef:
        D = len(tensor.get_shape())
        new_D = len(shape)
        assert D <= new_D

        new_shape = [1] * (new_D - D) + list(tensor.get_shape())
        new_strides = [1] * (new_D - D) + list(tensor.get_strides())
        for i in range(new_D):
            if new_shape[-i] != shape[-i]:
                assert new_shape[-i] == 1
                new_shape[-i] = shape[-i]
                new_strides[-i] = 0

        offset = tensor.get_offset()

        return tensor.create_view(offset, tuple(new_shape), tuple(new_strides))
