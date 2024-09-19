from __future__ import annotations

from copper.codegen import SlangCodeGen

from .base import IOFlag, IVariable, Wrapper, register_wrapper, wrap, register_assignment, assign
from ..reflection import ScalarKind, Modifier, SlangName, SlangType, ScalarType, InterfaceType, opaque_type
from ..reflection import is_flattenable, flatten

# from ..layers import TensorRef, tensor_layer, ReflectedType

import logging
import enum
import math

from typing import cast, Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..codegen import SlangCodeGen
    from ..layers import ShaderCursor

import torch


"""
class Device:
    pass

class CpuDevice(Device):
    pass

class GpuDevice(Device):
    def gpu_idx(self) -> int:
        raise NotImplementedError
    
class DXDevice(Device):
    pass

class CudaDevice:
    def stream(self):
        raise NotImplementedError
"""


class CudaDevice:
    def idx(self) -> int:
        raise NotImplementedError

    def stream(self):
        raise NotImplementedError


class TensorStorage:
    pass


class CpuTensorStorage(TensorStorage):
    def ptr(self):
        raise NotImplementedError

    # Size in bytes
    def size(self):
        raise NotImplementedError


class CudaTensorStorage(TensorStorage):
    def ptr(self):
        raise NotImplementedError

    # Size in bytes
    def size(self):
        raise NotImplementedError


class SlangStorageKind(enum.Enum):
    ReadOnly = enum.auto()
    ReadWrite = enum.auto()
    Atomic = enum.auto()

    def is_writable(self) -> bool:
        return self is SlangStorageKind.ReadWrite or self is SlangStorageKind.Atomic

    def is_atomic(self) -> bool:
        return self is SlangStorageKind.Atomic


class SlangTensorStorage(TensorStorage):
    def __init__(self, kind: SlangStorageKind, dtype: SlangType):
        super().__init__()
        self.kind = kind
        self.dtype = dtype

    def is_writable(self) -> bool:
        return self.kind.is_writable()

    def is_atomic(self) -> bool:
        return self.kind.is_atomic()

    def __str__(self):
        flags = "Grad" if self.is_atomic() else "RW" if self.is_writable() else ""
        return f"I{flags}TensorStorage<{self.dtype}>"


class ITensor(InterfaceType):
    def __init__(self, dtype: SlangType, ndim: int):
        super().__init__()
        self.dtype = dtype
        self.ndim = ndim

    def __str__(self):
        return f"ITensor<{self.dtype}, {self.ndim}>"


class IRWTensor(ITensor):
    pass


class ITensorLayout(InterfaceType):
    def __init__(self, ndim: int):
        super().__init__()
        self.ndim = ndim

    def __str__(self):
        return f"ITensorLayout<{self.ndim}>"


class ITensorStorage(InterfaceType):
    def __init__(self, element_type: SlangType):
        super().__init__()
        self.element_type = element_type

    def __str__(self):
        return f"ITensorStorage<{self.element_type}>"


class StridedLayout(ITensorLayout):  # TODO: StructType? StrIndexable
    def __init__(self, ndim: int):
        super().__init__(ndim)

    def __str__(self):
        return f"StridedLayout<{self.ndim}>"


class TensorWithStorage(ITensor):  # TODO: StructType? StrIndexable
    def __init__(self, dtype: SlangType, layout: ITensorLayout, storage: ITensorStorage):
        super().__init__(dtype, layout.ndim)
        self.layout = layout
        self.storage = storage

    def __str__(self):
        return f"TensorWithStorage<{self.dtype}, {self.ndim}, {self.layout}, {self.storage}>"


class DiffTensor(ITensor):  # TODO: StructType? StrIndexable
    def __init__(self, primal: ITensor, differential: ITensor):
        assert primal.dtype.differentiable()
        assert primal.dtype.differentiate() == differential.dtype
        assert primal.ndim == differential.ndim

        super().__init__(primal.dtype, primal.ndim)
        self.primal = primal
        self.differential = differential

    def __str__(self):
        return f"DiffTensor<{self.dtype}, {self.ndim}, {self.primal}, {self.differential}>"


class TorchCudaDevice(CudaDevice):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def idx(self):
        return self.device.index

    def stream(self):
        return torch.cuda.current_stream(self.device).cuda_stream


class TorchStorage(TensorStorage):
    def __init__(self, storage: torch.UntypedStorage):
        super().__init__()
        self.storage = storage

    def ptr(self):
        return self.storage.data_ptr()

    def size(self):
        return self.storage.size()


class TorchCudaStorage(TorchStorage, CudaTensorStorage):
    def __init__(self, storage: torch.UntypedStorage):
        assert storage.device.type == 'cuda'
        super().__init__(storage)


class TorchCpuStorage(TorchStorage, CpuTensorStorage):
    def __init__(self, storage: torch.UntypedStorage):
        assert storage.device.type == 'cpu'
        super().__init__(storage)


kind_to_dtype = {
    ScalarKind.Bool: torch.bool,
    ScalarKind.Uint8: torch.uint8,
    ScalarKind.Uint16: torch.int16,
    ScalarKind.Uint: torch.int32,
    ScalarKind.Uint64: torch.int64,
    ScalarKind.Int8: torch.int8,
    ScalarKind.Int16: torch.int16,
    ScalarKind.Int: torch.int32,
    ScalarKind.Int64: torch.int64,
    ScalarKind.Float16: torch.float16,
    ScalarKind.Float: torch.float32,
    ScalarKind.Float64: torch.float64,
}
dtype_to_kind = {v: k for k, v in kind_to_dtype.items()}


def convert_tensor(value: torch.Tensor) -> SlangTensor:
    if value.device.type == 'cuda':
        storage = TorchCudaStorage(value.untyped_storage())
    elif value.device.type == 'cpu':
        storage = TorchCpuStorage(value.untyped_storage())
    else:
        raise ValueError(f"Unsupported torch device {value.device}")

    dtype = ScalarType(dtype_to_kind[value.dtype])
    layout = StridedLayout(value.storage_offset(), value.stride())

    tensor = TensorWithStorage(dtype, tuple(value.shape), layout, storage)

    if value.grad is not None:
        return TensorWithGrads(tensor, convert_tensor(value.grad))
    else:
        return tensor


def assign_identical_tensor(type: ITensor, tensor: Tensor, flag: IOFlag):
    shape = batch_size
    if not self.is_scalar:
        shape += (math.prod(self.out_shape),)
    if input_tensor.is_empty():
        logging.debug(
            f"Creating indexed meta-tensor {self.in_dtype.to_slang_type()}{shape}"
        )
        meta = tensor_layer().meta_tensor(self.in_dtype, shape)
        input_tensor.point_to(meta)
        return input_tensor
    else:
        logging.debug(
            f"Broadcasting indexed tensor from {input_tensor.get_shape()} to {shape}"
        )
        result = TensorRef.broadcast(input_tensor, shape)
        if input_tensor.buffer() is None:
            input_tensor.point_to(result)
            result = input_tensor
        return result


class CastTensor(Wrapper):
    def batch_size(self, value: Any) -> tuple[int, ...]:
        return ()

    def broadcast(self, value: Any, batch_size: tuple[int, ...]) -> tuple[IVariable, Any]:
        return super().broadcast(value, batch_size)


class TensorRef:
    def empty(self) -> bool:
        raise NotImplementedError

    def get(self) -> Tensor:
        raise NotImplementedError

    def point_to(self, value: Tensor):
        raise NotImplementedError


class Tensor:
    def get_dtype(self) -> SlangType:
        raise NotImplementedError

    def get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def get_ndim(self) -> int:
        return len(self.get_shape())

    def get_type(self) -> ITensor:
        raise NotImplementedError


class TypedStorage:
    def dtype(self) -> SlangType:
        raise NotImplementedError

    def cast(self, dtype: SlangType) -> Optional[TypedStorage]:
        return None


class StridedTensor(Tensor):
    def __init__(self, storage: TypedStorage, shape: tuple[int, ...], strides: Optional[tuple[int, ...]] = None, offset: int = 0):
        super().__init__()

        if strides is None:
            strides = (1, )
            for d in reversed(shape[1:]):
                strides = (d * strides[0], ) + strides

        assert len(shape) == len(strides)

        self.storage = storage
        self.shape = shape
        self.strides = strides
        self.offset = offset

    def get_dtype(self) -> SlangType:
        return self.storage.dtype()

    def get_shape(self) -> tuple[int, ...]:
        return self.shape

    """
    def create_view(
        self, offset: int, shape: tuple[int, ...], strides: tuple[int, ...]
    ) -> StridedTensor:
        raise NotImplementedError

    def broadcast(self, shape: tuple[int, ...]) -> StridedTensor:
        D = self.get_ndim()
        new_D = len(shape)
        assert D <= new_D

        new_shape = [1] * (new_D - D) + list(self.get_shape())
        new_strides = [1] * (new_D - D) + list(self.get_strides())
        for i in range(new_D):
            if new_shape[-i] != shape[-i]:
                assert new_shape[-i] == 1
                new_shape[-i] = shape[-i]
                new_strides[-i] = 0

        offset = self.get_offset()

        return self.create_view(offset, tuple(new_shape), tuple(new_strides))"""

#    if flag.read() and tensor.buffer() is None:
#        raise ValueError(f"Can't pass empty TensorRef for in parameter")


def pass_tensor(value: Tensor) -> Wrapper:
    pass


class SlicedTensorType(IVariable):  # TODO
    def __init__(self, slice_dim: int, tensor: ITensor):
        super().__init__(ITensor(tensor.dtype, tensor.ndim - slice_dim))
        self.slice_dim = slice_dim
        self.tensor = tensor


class SlicedTensorWrapper(Wrapper):
    def __init__(self, slice_dim: int, tensor: Wrapper):
        super().__init__()
        self.slice_dim = slice_dim
        self.tensor = tensor

    def batch_size(self, value: Tensor) -> tuple[int, ...]:
        return value.get_shape()[:-self.slice_dim]

    def broadcast(self, value: Tensor, batch_size: tuple[int, ...]) -> tuple[IVariable, Any]:
        tensor_shape = batch_size + value.get_shape()[-self.slice_dim:]
        var, val = self.tensor.broadcast(value, tensor_shape)

        return SlicedTensorType(self.slice_dim, var), val


def slice_tensor(type: ITensor, value: Tensor, mod: Modifier):
    if type.dtype != value.get_dtype():
        raise ValueError(f"Tensor dtype ({value.get_dtype()}) must match "
                         f"slang tensor dtype ({type.dtype}")

    if value.get_ndim() < type.ndim:
        raise ValueError(f"Tensor has too few dimensions: Expected {type.ndim}, "
                         f"received {value.get_ndim()}")

    return SlicedTensorWrapper(type.ndim, pass_tensor(value))


@opaque_type("IndexedTensor", IVariable, [SlangType, int, ITensor])
class IndexedTensor(IVariable):
    def __init__(self, dtype: SlangType, ndim: int, tensor_type: ITensor):
        super().__init__(tensor_type.dtype)
        self.tensor_type = tensor_type


def assign_indexed_tensor(type: IndexedTensor, var: ShaderCursor, value: Tensor):
    assign(type.tensor_type, var["tensor"], value)


register_assignment(IndexedTensor, assign_indexed_tensor)


class IndexedTensorWrapper(Wrapper):
    def __init__(self, dtype: SlangType, type_shape: tuple[int, ...], elem_type: SlangType):
        super().__init__()
        self.dtype = dtype
        self.type_shape = type_shape
        self.elem_type = elem_type

    def batch_size(self, tensor: Tensor) -> tuple[int, ...]:
        return tensor.get_shape()[:-len(self.type_shape)]

    def broadcast(self, value: Tensor, batch_size: tuple[int, ...]) -> tuple[IVariable, Any]:
        var, val = self.tensor.broadcast(value, batch_size)
        assert isinstance(var, ITensor)
        assert isinstance(val, Tensor)

        return IndexedTensor(val.get_dtype(), val.get_ndim(), var), val


def index_tensor(dtype: SlangType, tensor: Tensor, flag: IOFlag):
    if dtype == tensor.get_dtype():
        return IndexedTensorWrapper(dtype, (), dtype)

    if not is_flattenable(dtype):
        raise ValueError(
            f"Tensor with dtype {tensor.get_dtype()} can't be converted to {dtype}")

    type_shape, elem_type = flatten(dtype)
    if elem_type != tensor.get_dtype():
        raise ValueError(f"Tensor with dtype {tensor.get_dtype()} can't be converted to "
                         f"{dtype}: Expecting dtype {elem_type}")

    shape = tensor.get_shape()
    batch_ndim = len(shape) - len(type_shape)
    if batch_ndim < 0:
        raise ValueError(f"Tensor with dtype {tensor.get_dtype()} can't be converted to "
                         f"{dtype}: Tensor has too few dimensions. Expected "
                         f"{len(type_shape)}, received {len(shape)}")

    if shape[batch_ndim:] != type_shape:
        raise ValueError(f"Tensor with dtype {tensor.get_dtype()} can't be converted to "
                         f"{dtype}: Expected a tensor with shape [..., "
                         f"{', '.join(str(d) for d in type_shape)}], but received "
                         f"a tensor with shape {list(shape)}")

    if isinstance(tensor, StridedTensor):
        is_contiguous = (
            tensor.strides[-1] == 1
            and all(
                tensor.strides[i] == tensor.strides[i + 1] * shape[i + 1]
                for i in range(batch_ndim, len(shape) - 1)
            )
        )

        if is_contiguous:
            cast_storage = tensor.storage.cast(ScalarType(elem_type))
            if cast_storage:
                type_ndim = len(type_shape)
                new_shape = tensor.shape[:-type_ndim]
                new_strides = tensor.strides[:-type_ndim]
                return StridedTensor(cast_storage, new_shape, new_strides, tensor.offset)

    # TODO
    raise ValueError("Blah bleh blih blah bluh")

    # return IndexedTensorWrapper()

    var.value = tensor

    if var.is_differentiable and False:
        kind = TensorKind.DiffRWTensor if var.is_out else TensorKind.DiffTensor
    else:
        kind = TensorKind.RWTensor if var.is_out else TensorKind.Tensor

    self.out_type = out_type
    self.out_shape = out_shape
    self.is_scalar = is_scalar
    super().__init__(kind, ScalarType(in_dtype), 0 if self.is_scalar else 1)


# Sequence of events:
#   - Given dst type and value, derive batch size
#   - Given dst type, value and batch size, derive src type
#   - Given dst type, value, batch size and src type, assign shader vars

class DeferredTensorWrapper(Wrapper):
    def __init__(self, dtype: SlangType):
        super().__init__()
        self.dtype = dtype

    def batch_size(self, value: Tensor) -> tuple[int, ...]:
        return ()

    def broadcast(self, ref: TensorRef, batch_size: tuple[int, ...]) -> tuple[IVariable, Any]:
        storage: TypedStorage  # TODO

        tensor = StridedTensor(storage, batch_size)
        ref.point_to(tensor)

        wrapper = wrap(IRWTensor(self.dtype, tensor.get_ndim()), tensor, IOFlag.Read)
        var, val = wrapper.broadcast(tensor, batch_size)

        assert isinstance(var, ITensor)

        return IndexedTensor(tensor.get_dtype(), tensor.get_ndim(), var), val


def wrap_tensor(type: SlangType, value: Tensor, flag: IOFlag):
    if isinstance(type, ITensor):
        if type.dtype == value.get_dtype():
            pass
        else:
            # slice tensor
            pass
    else:
        return index_tensor(type, value, flag)


class TensorKind(enum.Enum):
    Tensor = enum.auto()
    RWTensor = enum.auto()
    DiffTensor = enum.auto()
    DiffRWTensor = enum.auto()
    GradTensor = enum.auto()

    def to_slang_type(self):
        if self is TensorKind.Tensor:
            return "Tensor"
        if self is TensorKind.RWTensor:
            return "RWTensor"
        if self is TensorKind.DiffTensor:
            return "DiffTensor"
        if self is TensorKind.DiffRWTensor:
            return "DiffRWTensor"
        if self is TensorKind.GradTensor:
            return "GradTensor"
        raise RuntimeError("Invalid tensor kind")

    def writeable(self) -> bool:
        return (
            self is TensorKind.RWTensor
            or self is TensorKind.DiffRWTensor
            or self is TensorKind.GradTensor
        )

    def differentiable(self) -> bool:
        return self is TensorKind.DiffTensor or self is TensorKind.DiffRWTensor


@opaque_type("Tensor", "RWTensor", "DiffTensor")
@opaque_type("Tensor1D", "Tensor2D", "Tensor3D", "Tensor4D")
@opaque_type("RWTensor1D", "RWTensor2D", "RWTensor3D", "RWTensor4D")
@opaque_type("DiffTensor1D", "DiffTensor2D", "DiffTensor3D", "DiffTensor4D")
# @opaque_type('DiffRWTensor', 'DiffRWTensor1D', 'DiffRWTensor2D', 'DiffRWTensor3D', 'DiffRWTensor4D')
# @opaque_type('GradTensor')
class TensorType(SlangType):
    def __init__(self, kind: TensorKind, dtype: SlangType, ndim: int):
        super().__init__()
        self.kind = kind
        self.dtype = dtype
        self.ndim = ndim

    def to_slang_type(self) -> str:
        out = self.kind.to_slang_type()
        if self.ndim <= 4:
            out += f"{self.ndim}D"
        out += "<" + self.dtype.to_slang_type()
        if self.ndim > 4:
            out += f", {self.ndim}"
        out += ">"
        return out

    def assign_vars(self, var: AssignmentCursor, val: Any):
        if not isinstance(val, TensorRef):
            var.error("Expected a tensor argument")
        if isinstance(var.type, TensorType):
            var.set_translator(SlicedTensor(var, val))
        else:
            var.set_translator(IndexedTensor(var, val))

    @staticmethod
    def from_reflection(name: SlangName, type: ReflectedType) -> TensorType:
        args = type.generic_args()
        assert len(args) >= 1
        basename = name.base

        if any(basename.endswith(nd) for nd in ("1D", "2D", "3D", "4D")):
            ndim = int(basename[-2])
            basename = basename[:-2]
        else:
            assert len(args) >= 2 and isinstance(args[1], int)
            ndim = args[1]

        dtype = args[0]
        assert isinstance(dtype, SlangType)

        result = TensorType(TensorKind[basename], dtype, ndim)

        logging.debug(f"Reflected a TensorType {result.to_slang_type()}")

        return result


class SlicedTensor(TensorType, BatchedType, TypeTranslator):
    def __init__(self, var: AssignmentCursor, tensor: TensorRef):
        assert isinstance(var.type, TensorType)
        type = cast(TensorType, var.type)

        if tensor.is_empty():
            var.error(f"An empty TensorRef is not allowed here")

        in_dtype = flattened_type(type.dtype)
        assert in_dtype is not None
        if tensor.get_dtype() != in_dtype:
            var.error(
                f"Tensor dtype ({tensor.get_dtype()}) must match "
                f"dtype of slang tensor ({in_dtype}"
            )

        if len(tensor.get_shape()) < type.ndim:
            var.error(
                f"Tensor has too few dimensions: Expected {type.ndim}, "
                f"received {len(tensor.get_shape())}"
            )

        super().__init__(type.kind, ScalarType(in_dtype), type.ndim)
        self.out_type = type
        var.value = tensor

    def declare_variable(self, var: Variable, gen: SlangCodeGen):
        gen.emit(f"Batch{self.to_slang_type()} {var.name};")

    def read_variable(self, var: Variable, var_name: str) -> str:
        return var_name + ".slice(batchIdx.dims)"

    def write_variable(
        self, var: Variable, var_name: str, value: str, gen: SlangCodeGen
    ):
        assert self.kind.writeable()

    def infer_batch_size(self, input_tensor: TensorRef) -> tuple[int, ...]:
        return input_tensor.get_shape()[: -self.out_type.ndim]

    def broadcast(self, input_tensor: TensorRef, batch_size: tuple[int, ...]):
        shape = batch_size + input_tensor.get_shape()[-self.out_type.ndim:]
        logging.debug(
            f"Broadcasting sliced tensor ({self.out_type.ndim}D) from "
            f"{input_tensor.get_shape()} to {batch_size}"
        )
        result = TensorRef.broadcast(input_tensor, shape)
        if input_tensor.buffer() is None:
            input_tensor.point_to(result)
            result = input_tensor
        return result

    def __repr__(self):
        return f"Slice[{self.to_slang_type()}]"
