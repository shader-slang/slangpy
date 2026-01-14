# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import Any, Optional, cast
from numpy import ScalarType
from slangpy import DataType, BufferUsage, TypeReflection, DeviceType
import torch

from slangpy.core.native import (
    AccessType,
    CallContext,
    CallMode,
    Shape,
    TensorRef,
    NativeTorchTensorMarshall,
)
from slangpy.bindings.boundvariableruntime import BoundVariableRuntime
from slangpy.bindings.marshall import ReturnContext
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES
from slangpy.bindings import BindContext, BoundVariable, CodeGenBlock
from slangpy.builtin.tensor import TensorMarshall, is_nested_array
from slangpy import Buffer, ShaderObject, ShaderCursor
from slangpy.reflection.reflectiontypes import (
    SlangProgramLayout,
    SlangType,
    ScalarType,
    VectorType,
    MatrixType,
    TensorType,
    TensorAccess,
)
from slangpy.reflection.lookup import innermost_type
import slangpy.builtin.tensorcommon as spytc

ST = TypeReflection.ScalarType
_torch_to_scalar_type = {
    torch.int8: ST.int8,
    torch.int16: ST.int16,
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
    torch.int16: DataType.int16,
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


def get_storage(context: CallContext, element_count: int, struct_size: int) -> Buffer:
    return context.device.create_buffer(
        size=element_count * struct_size,
        struct_size=struct_size,
        usage=BufferUsage.shared | BufferUsage.unordered_access | BufferUsage.shader_resource,
    )


class TensorRefMarshall(TensorMarshall):
    def __init__(
        self,
        layout: SlangProgramLayout,
        torch_dtype: torch.dtype,
        slang_dtype: SlangType,
        dims: int,
        d_in: Optional["TensorRefMarshall"],
        d_out: Optional["TensorRefMarshall"],
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
        self.d_in: Optional[TensorRefMarshall]
        self.d_out: Optional[TensorRefMarshall]

        self.torch_dtype = torch_dtype
        self.slang_dtype = slang_dtype

    def get_shape(self, value: Optional[TensorRef] = None) -> Shape:
        if value is not None:
            return Shape(cast(torch.Tensor, value.tensor).shape)  # type: ignore
        else:
            return Shape((-1,) * self.dims)

    def create_calldata(
        self, context: CallContext, binding: "BoundVariableRuntime", data: TensorRef
    ) -> Any:
        if data.tensor is None:
            raise ValueError("Missing required tensor data")
        primal = cast(torch.Tensor, data.tensor)  # type: ignore

        data.last_access = binding.access
        shape = tuple(primal.shape)
        strides = primal.stride()

        bound_shape = shape[-len(binding.vector_type.shape) :]
        if any([b != -1 and a != b for a, b in zip(bound_shape, binding.vector_type.shape)]):  # type: ignore
            raise ValueError(
                f"Tensor shape {shape} does not match expected shape {binding.vector_type.shape}"
            )
        assert primal.is_cuda

        # For CUDA tensors, the C++ fast path handles marshalling directly
        # This Python method is only used for non-CUDA device interop
        if context.device.info.type != DeviceType.cuda:

            data_type = _torch_to_data_type[self.torch_dtype]

            # For empty tensors, create a minimal placeholder buffer
            # Shaders still need a valid buffer binding even if no data is accessed
            element_count = max(1, primal.numel())
            data.interop_buffer = get_storage(context, element_count, primal.element_size())

            # Only copy data if tensor has elements
            if primal.numel() > 0:
                interop_tensor = cast(
                    torch.Tensor,
                    data.interop_buffer.to_torch(type=data_type, shape=shape, strides=strides),
                )
                interop_tensor.copy_(primal)

            primal_calldata = {
                "_data": data.interop_buffer,
                "_offset": 0,
                "_strides": strides,
                "_shape": shape,
            }

            if not self.d_in and not self.d_out:
                return primal_calldata

            result = {"_primal": primal_calldata}
            if self.d_in is not None:
                if data.grad_in is None:
                    raise ValueError("Missing required input gradients")
                result["_grad_in"] = self.d_in.create_calldata(context, binding, data.grad_in)
            if self.d_out is not None:
                if data.grad_out is None:
                    raise ValueError("Missing tensor to hold output gradients")
                result["_grad_out"] = self.d_out.create_calldata(context, binding, data.grad_out)

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
        else:
            # CUDA tensors are handled by C++ fast path - this should not be reached
            raise RuntimeError(
                "CUDA tensors should be handled by C++ fast path, not Python marshalling"
            )

    def read_calldata(
        self,
        context: CallContext,
        binding: "BoundVariableRuntime",
        data: TensorRef,
        result: Any,
    ):
        if context.device.info.type != DeviceType.cuda:
            assert data.tensor is not None
            assert data.interop_buffer is not None
            primal = cast(torch.Tensor, data.tensor)  # type: ignore

            # Only copy data back for non-empty tensors
            if primal.numel() > 0:
                shape = tuple(primal.shape)
                strides = primal.stride()
                data_type = _torch_to_data_type[self.torch_dtype]
                interop_tensor = cast(
                    torch.Tensor,
                    data.interop_buffer.to_torch(type=data_type, shape=shape, strides=strides),
                )

                primal.untyped_storage().copy_(interop_tensor.untyped_storage())

            data.interop_buffer = None

            if self.d_in is not None:
                assert data.grad_in is not None
                self.d_in.read_calldata(context, binding, data.grad_in, result["_grad_in"])
            if self.d_out is not None:
                assert data.grad_out is not None
                self.d_out.read_calldata(context, binding, data.grad_out, result["_grad_out"])

    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        # Overall shape of tensor must contain the call, plus the shape of the slang datatype
        # i.e. if a float tensor is to store 4x4 matrix results, it needs the shape to be
        # extended by (4,4)
        combined_shape = context.call_shape.as_tuple() + self.slang_dtype.shape.as_tuple()
        return TensorRef(
            -1, torch.empty(combined_shape, dtype=self.torch_dtype, device=torch.device("cuda"))
        )

    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: Any) -> Any:
        return data


class TorchTensorMarshall(NativeTorchTensorMarshall):
    """
    Marshall for raw torch.Tensor objects (not wrapped in TensorRef).

    Inherits from NativeTorchTensorMarshall which provides:
    - Fast native get_shape via TorchBridge
    - Native write_shader_cursor_pre_dispatch for CUDA tensors

    This class adds:
    - Type resolution for binding
    - Code generation for kernels
    - Shader object building
    """

    def __init__(
        self,
        layout: SlangProgramLayout,
        torch_dtype: torch.dtype,
        slang_dtype: SlangType,
        dims: int,
        d_in: Optional["TorchTensorMarshall"],
        d_out: Optional["TorchTensorMarshall"],
    ):
        # Validate element type
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

        # Determine writability and tensor type
        writable = True  # Torch tensors are always potentially writable
        has_derivatives = d_in is not None or d_out is not None

        # Get the slang tensor type
        slang_type = layout.tensor_type(
            element_type=dtype,
            dims=full_dims,
            access=TensorAccess.read_write if writable else TensorAccess.read,
            tensor_type=TensorType.difftensor if has_derivatives else TensorType.tensor,
        )

        if not slang_type:
            raise ValueError(
                f"Failed to find tensor type to contain element {dtype.full_name}. "
                f"If using differentiable tensors, this can imply that the element type "
                f"does not support both the IDifferentiable and IAtomicAddable interfaces."
            )

        # Store for Python-side use
        self._layout = layout
        self._torch_dtype = torch_dtype
        self._slang_dtype = slang_dtype

        # Initialize base class (sets d_in, d_out, dims, writable etc in C++)
        super().__init__(
            dims=full_dims,
            writable=writable,
            slang_type=slang_type,
            slang_element_type=dtype,
            element_layout=dtype.buffer_layout.reflection,
            d_in=d_in,
            d_out=d_out,
        )

    @property
    def layout(self) -> SlangProgramLayout:
        return self._layout

    @property
    def torch_dtype(self) -> torch.dtype:
        return self._torch_dtype

    @property
    def slang_dtype(self) -> SlangType:
        return self._slang_dtype

    def __repr__(self) -> str:
        return f"TorchTensor[dtype={self.slang_element_type.full_name}, dims={self.dims}, writable={self.writable}]"

    @property
    def has_derivative(self) -> bool:
        return self.d_in is not None or self.d_out is not None

    @property
    def is_writable(self) -> bool:
        return self.writable

    def resolve_types(self, context: BindContext, bound_type: SlangType):
        """Resolve types during binding phase."""
        return spytc.resolve_types(self, context, bound_type)

    def reduce_type(self, context: BindContext, dimensions: int):
        """Reduce tensor type by consuming dimensions."""
        return spytc.reduce_type(self, context, dimensions)

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: SlangType,
    ):
        """Resolve dimensionality during vectorization."""
        return spytc.resolve_dimensionality(self, context, binding, vector_target_type)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        """Generate call data code for the kernel."""
        return spytc.gen_calldata(self, cgb, context, binding)

    def build_shader_object(self, context: BindContext, data: torch.Tensor) -> ShaderObject:
        """Build shader object for dispatch."""
        so = context.device.create_shader_object(self.slang_type.uniform_layout.reflection)
        cursor = ShaderCursor(so)

        if not self.has_derivative:
            # Simple case - just write the tensor uniforms
            cursor.write(self._get_tensor_uniforms(data))
        else:
            # Differentiated case - not yet supported for raw tensors
            raise NotImplementedError("Gradient support for raw torch.Tensor not yet implemented")

        return so

    def _get_tensor_uniforms(self, tensor: torch.Tensor) -> dict[str, Any]:
        """Extract uniform data from a torch tensor."""
        return {
            "_data": tensor.data_ptr(),
            "_shape": list(tensor.shape),
            "_strides": list(tensor.stride()),
            "_offset": 0,
        }


def create_torch_tensor_marshall(layout: SlangProgramLayout, value: Any):
    """Factory function for creating TorchTensorMarshall for raw torch.Tensor."""
    if isinstance(value, ReturnContext):
        slang_dtype = value.slang_type
        torch_dtype = _slang_dtype_to_torch(innermost_type(slang_dtype))
        if torch_dtype is None:
            raise ValueError(f"Unsupported slang type {value.slang_type}")
        return TorchTensorMarshall(
            layout,
            torch_dtype,
            slang_dtype,
            value.bind_context.call_dimensionality,
            None,
            None,
        )
    elif isinstance(value, torch.Tensor):
        torch_dtype = value.dtype
        slang_dtype = _torch_dtype_to_slang(torch_dtype, layout)
        if slang_dtype is None:
            raise ValueError(f"Unsupported torch dtype {value.dtype}")
        # No gradient support for raw tensors yet
        return TorchTensorMarshall(
            layout,
            torch_dtype,
            slang_dtype,
            len(value.shape),
            None,
            None,
        )
    else:
        raise ValueError(f"Type {type(value)} is unsupported for torch.Tensor marshall")


def create_tensor_marshall(layout: SlangProgramLayout, value: Any):
    if isinstance(value, ReturnContext):
        if value.bind_context.call_dimensionality == 0 and False:
            return tr.get_or_create_type(layout, ValueRef, value)
        else:
            slang_dtype = value.slang_type
            torch_dtype = _slang_dtype_to_torch(innermost_type(slang_dtype))
            if torch_dtype is None:
                raise ValueError(f"Unsupported slang type {value.slang_type}")
            marshall = TensorRefMarshall(
                layout,
                torch_dtype,
                slang_dtype,
                value.bind_context.call_dimensionality,
                None,
                None,
            )
    elif isinstance(value, TensorRef):
        assert value.tensor is not None
        torch_dtype = value.tensor.dtype
        slang_dtype = _torch_dtype_to_slang(torch_dtype, layout)
        if slang_dtype is None:
            raise ValueError(f"Unsupported torch dtype {value.tensor.dtype}")

        d_in = create_tensor_marshall(layout, value.grad_in) if value.grad_in is not None else None
        d_out = (
            create_tensor_marshall(layout, value.grad_out) if value.grad_out is not None else None
        )

        marshall = TensorRefMarshall(
            layout, torch_dtype, slang_dtype, len(value.tensor.shape), d_in, d_out
        )
    else:
        raise ValueError(f"Type {type(value)} is unsupported for torch.Tensor marshall")

    return marshall


def hash_tensor(value: Any) -> str:
    raise ValueError(f"TensorRef should not need a hash key as it is native object")


def hash_torch_tensor(value: Any) -> str:
    raise ValueError(f"torch.Tensor should not need a hash key as it is native object")


# Register TensorRef handlers (legacy)
PYTHON_TYPES[TensorRef] = create_tensor_marshall
PYTHON_SIGNATURES[TensorRef] = hash_tensor

# Register torch.Tensor handlers (new)
PYTHON_TYPES[torch.Tensor] = create_torch_tensor_marshall
PYTHON_SIGNATURES[torch.Tensor] = hash_torch_tensor
