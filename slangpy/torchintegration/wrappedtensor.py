# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import Any, Optional, cast
from numpy import ScalarType
from slangpy import DataType, Device, BufferUsage, TypeReflection
import torch

from slangpy.core.native import AccessType, CallContext, CallMode, Shape
from slangpy.bindings.boundvariableruntime import BoundVariableRuntime
from slangpy.bindings.marshall import ReturnContext
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES
from slangpy.builtin.tensor import TensorMarshall, is_nested_array
from slangpy import Buffer
from slangpy.reflection.reflectiontypes import (
    SlangProgramLayout,
    SlangType,
    ScalarType,
    VectorType,
    MatrixType,
)
from slangpy.types.buffer import innermost_type

ST = TypeReflection.ScalarType
_torch_to_scalar_type = {
    torch.int8: ST.int8,
    torch.int32: ST.int16,
    torch.int32: ST.int32,
    torch.int64: ST.int64,
    torch.uint8: ST.uint8,
    torch.float16: ST.float16,
    torch.float32: ST.float32,
    torch.float64: ST.float64,
}
_scalar_type_to_torch = {y: x for x, y in _torch_to_scalar_type.items()}
_torch_to_data_type = {
    torch.int8: DataType.int8,
    torch.int32: DataType.int16,
    torch.int32: DataType.int32,
    torch.int64: DataType.int64,
    torch.uint8: DataType.uint8,
    torch.float16: DataType.float16,
    torch.float32: DataType.float32,
    torch.float64: DataType.float64,
}


def _slang_dtype_to_torch(slang_dtype: SlangType) -> Optional["torch.dtype"]:
    if isinstance(slang_dtype, ScalarType):
        return _scalar_type_to_torch.get(slang_dtype.slang_scalar_type)
    return None


def _torch_dtype_to_slang(
    torch_dtype: "torch.dtype", layout: SlangProgramLayout
) -> Optional[SlangType]:
    scalar_type = _torch_to_scalar_type.get(torch_dtype)
    if scalar_type is None:
        return None
    return layout.scalar_type(scalar_type)


GLOBAL_STORAGE: dict[tuple[Device, int, int], list[Buffer]] = {}


def get_or_create_storage(context: CallContext, element_count: int, struct_size: int) -> Buffer:
    value = (context.device, element_count, struct_size)
    if value not in GLOBAL_STORAGE:
        GLOBAL_STORAGE[value] = []
    if len(GLOBAL_STORAGE[value]) == 0:
        return context.device.create_buffer(
            size=element_count * struct_size,
            struct_size=struct_size,
            usage=BufferUsage.shared | BufferUsage.unordered_access | BufferUsage.shader_resource,
        )
    return GLOBAL_STORAGE[value].pop()


def return_storage(context: CallContext, buffer: Buffer):
    value = (
        context.device,
        buffer.desc.size // buffer.desc.struct_size,
        buffer.struct_size,
    )
    GLOBAL_STORAGE[value].append(buffer)


class WrappedTensor:
    def __init__(self, primal: Optional[torch.Tensor] = None, id: int = -1):
        super().__init__()

        self.id = id
        self.primal = primal
        self.grad_in: Optional[WrappedTensor] = None
        self.grad_out: Optional[WrappedTensor] = None
        self.last_access_type: tuple[AccessType, AccessType] = (
            AccessType.none,
            AccessType.none,
        )
        self.temp_storage_tensor: Optional[torch.Tensor] = None
        self.temp_storage_buffer: Optional[Buffer] = None

    @property
    def shape(self):
        return Shape(self.primal.shape) if self.primal is not None else Shape()

    def collect_streams(self, streams: set[int], include_meta: bool):
        if self.primal is not None and (
            self.primal.is_cuda or (self.primal.is_meta and include_meta)
        ):
            device = self.primal.device if self.primal.is_cuda else None
            stream = torch.cuda.current_stream(device).cuda_stream
            streams.add(stream)
        if self.grad_in:
            self.grad_in.collect_streams(streams, include_meta)
        if self.grad_out:
            self.grad_out.collect_streams(streams, include_meta)


class WrappedTensorMarshall(TensorMarshall):
    def __init__(
        self,
        layout: SlangProgramLayout,
        torch_dtype: torch.dtype,
        slang_dtype: SlangType,
        dims: int,
        d_in: Optional["WrappedTensorMarshall"],
        d_out: Optional["WrappedTensorMarshall"],
    ):

        dtype = innermost_type(slang_dtype)
        can_convert = (
            is_nested_array(slang_dtype)
            or isinstance(slang_dtype, ScalarType)
            or isinstance(slang_dtype, VectorType)
            or isinstance(slang_dtype, MatrixType)
        )
        if not can_convert or len(slang_dtype.shape) > 2:
            raise ValueError(f"Torch tensors do not support data type {slang_dtype.full_name}")

        full_dims = dims + len(slang_dtype.shape)

        super().__init__(layout, dtype, full_dims, True, d_in, d_out)
        self.d_in: Optional[WrappedTensorMarshall]
        self.d_out: Optional[WrappedTensorMarshall]

        self.torch_dtype = torch_dtype
        self.slang_dtype = slang_dtype

    def get_shape(self, value: Optional[WrappedTensor] = None) -> Shape:
        if value is not None:
            return Shape(value.primal.shape)  # type: ignore
        else:
            return Shape((-1,) * self.dims)

    def create_calldata(
        self, context: CallContext, binding: "BoundVariableRuntime", data: WrappedTensor
    ) -> Any:
        if data.primal is None:
            raise ValueError("Missing required tensor data")
        data.last_access_type = binding.access

        shape = tuple(data.primal.shape)
        offset = data.primal.storage_offset()
        strides = data.primal.stride()

        bound_shape = shape[-len(binding.vector_type.shape) :]
        if any([b != -1 and a != b for a, b in zip(bound_shape, binding.vector_type.shape)]):  # type: ignore
            raise ValueError(
                f"Tensor shape {shape} does not match expected shape {binding.vector_type.shape}"
            )

        assert data.primal.is_cuda

        data_type = _torch_to_data_type[self.torch_dtype]
        temp_storage = get_or_create_storage(
            context, data.primal.numel(), data.primal.element_size()
        )
        temp_storage_tensor = cast(
            torch.Tensor,
            temp_storage.to_torch(type=data_type, shape=shape, strides=strides),
        )
        temp_storage_tensor.copy_(data.primal)

        data.temp_storage_buffer = temp_storage
        data.temp_storage_tensor = temp_storage_tensor

        temp_storage_tensor.untyped_storage().copy_(
            data.primal.untyped_storage(), non_blocking=False
        )

        primal_calldata = {
            "buffer": temp_storage,
            "layout": {"offset": offset, "strides": strides},
            "_shape": shape,
        }

        if not self.d_in and not self.d_out:
            return primal_calldata

        result = {"primal": primal_calldata}
        if self.d_in is not None:
            if data.grad_in is None:
                raise ValueError("Missing required input gradients")
            result["d_in"] = self.d_in.create_calldata(context, binding, data.grad_in)
        if self.d_out is not None:
            if data.grad_out is None:
                raise ValueError("Missing tensor to hold output gradients")
            result["d_out"] = self.d_out.create_calldata(context, binding, data.grad_out)

        if (
            context.call_mode != CallMode.prim
            and data.grad_in is not None
            and data.grad_in is data.grad_out
        ):
            if binding.access[1] == AccessType.readwrite:
                raise ValueError(
                    "inout parameter gradients need separate buffers for inputs and outputs (see Tensor.with_grads)"
                )

        return result

    def read_calldata(
        self,
        context: CallContext,
        binding: "BoundVariableRuntime",
        data: WrappedTensor,
        result: Any,
    ):
        assert data.primal is not None
        assert data.temp_storage_tensor is not None
        assert data.temp_storage_buffer is not None
        data.primal.untyped_storage().copy_(data.temp_storage_tensor.untyped_storage())
        return_storage(context, data.temp_storage_buffer)
        data.temp_storage_buffer = None
        data.temp_storage_tensor = None
        if self.d_in is not None:
            assert data.grad_in is not None
            self.d_in.read_calldata(context, binding, data.grad_in, result["d_in"])
        if self.d_out is not None:
            assert data.grad_out is not None
            self.d_out.read_calldata(context, binding, data.grad_out, result["d_out"])

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        # Overall shape of tensor must contain the call, plus the shape of the slang datatype
        # i.e. if a float tensor is to store 4x4 matrix results, it needs the shape to be
        # extended by (4,4)
        combined_shape = context.call_shape.as_tuple() + self.slang_dtype.shape.as_tuple()
        return WrappedTensor(
            torch.empty(combined_shape, dtype=self.torch_dtype, device=torch.device("cuda"))
        )

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: Any) -> Any:
        return data


def create_tensor_marshall(layout: SlangProgramLayout, value: Any):
    if isinstance(value, ReturnContext):
        if value.bind_context.call_dimensionality == 0 and False:
            return tr.get_or_create_type(layout, ValueRef, value)
        else:
            slang_dtype = value.slang_type
            torch_dtype = _slang_dtype_to_torch(innermost_type(slang_dtype))
            if torch_dtype is None:
                raise ValueError(f"Unsupported slang type {value.slang_type}")
            marshall = WrappedTensorMarshall(
                layout,
                torch_dtype,
                slang_dtype,
                value.bind_context.call_dimensionality,
                None,
                None,
            )
    elif isinstance(value, WrappedTensor):
        assert value.primal is not None
        torch_dtype = value.primal.dtype
        slang_dtype = _torch_dtype_to_slang(torch_dtype, layout)
        if slang_dtype is None:
            raise ValueError(f"Unsupported torch dtype {value.primal.dtype}")

        d_in = create_tensor_marshall(layout, value.grad_in) if value.grad_in is not None else None
        d_out = (
            create_tensor_marshall(layout, value.grad_out) if value.grad_out is not None else None
        )

        marshall = WrappedTensorMarshall(
            layout, torch_dtype, slang_dtype, len(value.primal.shape), d_in, d_out
        )
    else:
        raise ValueError(f"Type {type(value)} is unsupported for torch.Tensor marshall")

    return marshall


def hash_tensor(value: Any) -> str:
    if isinstance(value, WrappedTensor):
        sig = f"TorchTensorWithGrad[hash_tensor(value.primal)"
        if value.grad_in is not None:
            sig += hash_tensor(value.grad_in)
        sig += ","
        if value.grad_out is not None:
            sig += hash_tensor(value.grad_out)
        sig += "]"

        return sig
    else:
        raise ValueError(f"Unexpected type {type(value).__name__} for tensor hashing")


PYTHON_TYPES[WrappedTensor] = create_tensor_marshall
PYTHON_SIGNATURES[WrappedTensor] = hash_tensor


def error_tensor_marshall(layout: SlangProgramLayout, value: Any):
    raise ValueError(
        f"torch.Tensor types can not be directly passed to SlangPy. Either use the \
                     pytorch integration (via TorchModule/TorchStruct/TorchFunction) or use a SlangPy \
                     tensor type."
    )


PYTHON_TYPES[torch.Tensor] = error_tensor_marshall
