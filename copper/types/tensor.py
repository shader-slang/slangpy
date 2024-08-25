from __future__ import annotations

from .interfaces import SlangType, BatchedType, TypeTranslator, opaque_type
from .typeutil import flattened_type, type_to_shape
from .arithmetic import ScalarType
from .base import SlangName

from ..layers import TensorRef, tensor_layer, ReflectedType

import logging
import enum
import math

from typing import cast, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..variable import AssignmentCursor, Variable
    from ..codegen import SlangCodeGen


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
        shape = batch_size + input_tensor.get_shape()[-self.out_type.ndim :]
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


class IndexedTensor(TensorType, BatchedType, TypeTranslator):
    def __init__(self, var: AssignmentCursor, tensor: TensorRef):
        if var.is_in and tensor.buffer() is None:
            var.error(f"Can't pass empty TensorRef for in parameter")

        in_dtype = flattened_type(var.type)
        if in_dtype is None:
            var.error(
                f"Parameter type {var.type.to_slang_type()} "
                "is not compatible with tensors"
            )

        out_type = var.type
        out_shape = type_to_shape(out_type)
        is_scalar = len(out_shape) == 0
        if not tensor.is_empty():
            dtype = tensor.get_dtype()
            if dtype != in_dtype:
                var.error(
                    f"Tensor dtype ({dtype}) must match corresponding slang type "
                    f"(expected {in_dtype} for slang type {out_type.to_slang_type()})"
                )

            shape = tensor.get_shape()
            strides = tensor.get_strides()
            batch_ndim = len(shape) - len(out_shape)
            if batch_ndim < 0:
                var.error(
                    f"Tensor has too few dimensions: {out_type.to_slang_type()} expected "
                    f"{len(out_shape)}, received {len(shape)}"
                )
            if out_shape != shape[batch_ndim:]:
                var.error(
                    f'Expected a tensor with shape [..., {", ".join(str(d) for d in out_shape)}] '
                    f"for parameter type {out_type.to_slang_type()}, "
                    f"but received a tensor of shape {list(shape)} instead"
                )

            is_contiguous = (
                is_scalar
                or strides[-1] == 1
                and all(
                    strides[i] == strides[i + 1] * shape[i + 1]
                    for i in range(batch_ndim, len(shape) - 1)
                )
            )
            if not is_contiguous:
                if var.is_out:
                    var.error("Tensors passed to out parameters must be contiguous")

                logging.debug(
                    f"IndexedTensor: Tensor with geometry "
                    f"shape{shape}, strides{strides} was made contiguous"
                )
                tensor = tensor.make_contiguous()
                strides = tensor.get_strides()

            if not is_scalar:
                shape = shape[:batch_ndim] + (math.prod(shape[batch_ndim:]),)
                strides = strides[:batch_ndim] + (1,)

            logging.debug(f"IndexedTensor: Mapped shape {shape} to new shape {shape}")

            tensor = tensor.create_view(0, shape, strides)
        var.value = tensor

        if var.is_differentiable and False:
            kind = TensorKind.DiffRWTensor if var.is_out else TensorKind.DiffTensor
        else:
            kind = TensorKind.RWTensor if var.is_out else TensorKind.Tensor

        self.out_type = out_type
        self.out_shape = out_shape
        self.is_scalar = is_scalar
        super().__init__(kind, ScalarType(in_dtype), 0 if self.is_scalar else 1)

    def declare_variable(self, var: Variable, gen: SlangCodeGen):
        gen.emit(f"Batch{self.to_slang_type()} {var.name};")

    def read_variable(self, var: Variable, var_name: str) -> str:
        result = var_name
        if not self.is_scalar:
            result += f".unflatten<{self.out_type.to_slang_type()}>()"
        return result + ".get(batchIdx.dims)"

    def write_variable(
        self, var: Variable, var_name: str, value: str, gen: SlangCodeGen
    ):
        assert self.kind.writeable()
        element = var_name
        if not self.is_scalar:
            element += f".unflatten<{self.out_type.to_slang_type()}>()"
        gen.emit(f"{element}.set(batchIdx.dims, {value});")

    def infer_batch_size(self, input_tensor: TensorRef) -> tuple[int, ...]:
        if input_tensor.is_empty():
            return ()
        elif self.is_scalar:
            return input_tensor.get_shape()
        else:
            return input_tensor.get_shape()[:-1]

    def broadcast(self, input_tensor: TensorRef, batch_size: tuple[int, ...]):
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

    def __repr__(self):
        return f"Batched[{self.out_type.to_slang_type()}]"
