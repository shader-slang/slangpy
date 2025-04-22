# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from slangpy.backend import Device, Buffer, BufferUsage, TypeReflection, CommandBuffer
from slangpy.core.utils import shape_to_contiguous_strides
from slangpy.reflection import SlangType, ScalarType, SlangProgramLayout
from slangpy.reflection import reflectiontypes
from slangpy.core.native import Shape
from slangpy.core.shapes import TShapeOrTuple
from slangpy.types.buffer import get_lookup_module, resolve_element_type, resolve_program_layout
from slangpy.core.native import Shape, NativeTensor, NativeTensorDesc

from warnings import warn

from typing import Optional, Any, cast, TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
    import torch

ST = TypeReflection.ScalarType
_numpy_to_sgl = {
    'int8': ST.int8, 'int16': ST.int16, 'int32': ST.int32, 'int64': ST.int64,
    'uint8': ST.uint8, 'uint16': ST.uint16, 'uint32': ST.uint32, 'uint64': ST.uint64,
    'float16': ST.float16, 'float32': ST.float32, 'float64': ST.float64
}
_sgl_to_numpy = {
    y: x for x, y in _numpy_to_sgl.items()
}


def innermost_type(slang_type: SlangType) -> SlangType:
    while True:
        if slang_type.element_type is not None and slang_type.element_type is not slang_type:
            slang_type = slang_type.element_type
        else:
            return slang_type


def _slang_to_numpy(slang_dtype: SlangType):
    elem_type = innermost_type(slang_dtype)
    if isinstance(elem_type, ScalarType) and elem_type.slang_scalar_type in _sgl_to_numpy:
        return np.dtype(_sgl_to_numpy[elem_type.slang_scalar_type])
    return None


def _numpy_to_slang(np_dtype: np.dtype[Any], device: Device) -> Optional[SlangType]:
    name = np_dtype.base.name
    if name not in _numpy_to_sgl:
        return None
    slang_dtype = reflectiontypes.scalar_names[_numpy_to_sgl[name]]
    if np_dtype.ndim > 0:
        for dim in reversed(np_dtype.shape):
            slang_dtype += f'[{dim}]'

    return get_lookup_module(device).find_type_by_name(slang_dtype)


class Tensor(NativeTensor):
    """
    Represents an N-D view of an underlying buffer with given shape and element type,
    and has optional gradient information attached. Element type must be differentiable.

    Strides and offset can optionally be specified and are given in terms of elements, not bytes.
    If omitted, a dense N-D grid is assumed (row-major).
    """

    def __init__(self, storage: Buffer, dtype: SlangType, shape: TShapeOrTuple,
                 strides: Optional[TShapeOrTuple] = None, offset: int = 0,
                 grad_in: Optional[Tensor] = None, grad_out: Optional[Tensor] = None):

        # Setup shape and stride.
        shape = Shape(shape)
        if strides is None:
            strides = shape.calc_contiguous_strides()
        if len(strides) != len(shape):
            raise ValueError("Number of strides must match number of dimensions")

        # Fill out native descriptor and initialize.
        desc = NativeTensorDesc()
        desc.shape = Shape(shape)
        desc.strides = Shape(strides)
        desc.offset = offset
        desc.dtype = dtype
        desc.offset = offset
        desc.element_layout = dtype.buffer_layout.reflection
        super().__init__(desc, storage, grad_in, grad_out)

        # Fix up some typing info
        self.dtype: SlangType
        self.grad_in: Optional[Tensor]
        self.grad_out: Optional[Tensor]

    def broadcast_to(self, shape: TShapeOrTuple):
        """
        Returns a new view of the tensor with the requested shape, following standard broadcasting rules.
        """
        return super().broadcast_to(Shape(shape))

    def view(self, shape: TShapeOrTuple, strides: TShapeOrTuple = Shape(), offset: int = 0):
        """
        Returns a new view of the tensor with the requested shape, strides and offset
        The offset is in elements (not bytes) and is specified relative to the current offset
        """
        return super().view(Shape(shape), Shape(strides), offset)

    def __str__(self):
        return str(self.to_numpy())

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """
        Copies tensor data into a numpy array with the same shape and strides. If the element type
        of the tensor is representable in numpy (e.g. floats, ints, arrays/vectors thereof), the
        ndarray will have a matching dtype. If the element type can't be represented in numpy (e.g. structs),
        the ndarray will be an array over the bytes of the buffer elements

        Examples:
        Tensor of dtype float3 with shape (4, 5)
            -> ndarray of dtype np.float32 with shape (4, 5, 3)
        Tensor of dtype struct Foo {...} with shape (5, )
            -> ndarray of dtype np.uint8 with shape (5, sizeof(Foo))
        """
        return cast(np.ndarray[Any, Any], super().to_numpy())

    def to_torch(self) -> 'torch.Tensor':
        """
        Returns a view of the buffer data as a torch tensor with the same shape and strides.
        See to_numpy for notes on dtype conversion
        """
        return cast('torch.Tensor', super().to_torch())

    def with_grads(self, grad_in: Optional[Tensor] = None, grad_out: Optional[Tensor] = None, zero: bool = False):
        """
        Returns a new tensor view with gradients attached. If called with no arguments, the
        tensor defaults to attaching a zeros-like initialized gradient tensor for both input and 
        output gradients.

        Specifying input gradients (grad_in) and/or output gradients (grad_out) allows more precise
        control over the gradient tensors, and is key when using a function that has inout parameters,
        so will want to both read and write gradients without causing race conditions. 

        When differentiating a slang call that wrote results to a tensor, gradients of the output will
        be read from grad_in (if not None). When differentiating a slang call that read inputs from a
        tensor, input gradients will be written to grad_out (if not None).
        """
        return super().with_grads(grad_in, grad_out, zero)

    def clear(self, command_buffer: Optional[CommandBuffer] = None):
        """
        Fill the tensor with zeros. If no command buffer is provided, a new one is created and
        immediately submitted. If a command buffer is provided the clear is simply appended to it
        but not automatically submitted.
        """
        super().clear()

    @staticmethod
    def numpy(device: Device, ndarray: np.ndarray[Any, Any]) -> Tensor:
        warn('Tensor.numpy is deprecated. Use Tensor.from_numpy instead.',
             DeprecationWarning, stacklevel=2)
        return Tensor.from_numpy(device, ndarray)

    @staticmethod
    def from_numpy(device: Device, ndarray: np.ndarray[Any, Any]) -> Tensor:
        """
        Creates a new tensor with the same contents, shape and strides as the given numpy array.
        """

        dtype = _numpy_to_slang(ndarray.dtype, device)
        if dtype is None:
            raise ValueError(f"Unsupported numpy dtype {ndarray.dtype}")
        if (ndarray.nbytes % ndarray.itemsize) != 0:
            raise ValueError(f"Unsupported numpy array")
        for stride in ndarray.strides:
            if (stride % ndarray.itemsize) != 0:
                raise ValueError(f"Unsupported numpy array")

        N = ndarray.nbytes // ndarray.itemsize
        flattened = np.lib.stride_tricks.as_strided(ndarray, (N, ), (ndarray.itemsize, ))
        strides = tuple(stride // ndarray.itemsize for stride in ndarray.strides)

        usage = BufferUsage.shader_resource | BufferUsage.unordered_access
        buffer = device.create_buffer(ndarray.nbytes, usage=usage, data=flattened)

        return Tensor(buffer, dtype, tuple(ndarray.shape), strides)

    @staticmethod
    def empty(device: Device, shape: TShapeOrTuple, dtype: Any, program_layout: Optional[SlangProgramLayout] = None) -> Tensor:
        """
        Creates a tensor with the requested shape and element type without attempting to initialize the data.
        """
        # If dtype supplied is not a SlangType, resolve it using the same mechanism as NDBuffer
        if not isinstance(dtype, SlangType):
            program_layout = resolve_program_layout(device, dtype, program_layout)
            dtype = resolve_element_type(program_layout, dtype)

        shape_tuple = shape if isinstance(shape, tuple) else shape.as_tuple()
        num_elems = math.prod(shape_tuple)

        usage = BufferUsage.shader_resource | BufferUsage.unordered_access
        buffer = device.create_buffer(element_count=num_elems,
                                      struct_size=dtype.buffer_layout.stride,
                                      usage=usage)

        return Tensor(buffer, dtype, shape)

    @staticmethod
    def zeros(device: Device, shape: TShapeOrTuple, dtype: Any) -> Tensor:
        """
        Creates a zero-initialized tensor with the requested shape and element type.
        """
        tensor = Tensor.empty(device, shape, dtype)
        tensor.clear()
        return tensor

    @staticmethod
    def empty_like(other: Tensor) -> Tensor:
        """
        Creates a new tensor with the same shape and element type as the given tensor, without initializing the data.
        """
        return Tensor.empty(other.storage.device, other.shape, other.dtype)

    @staticmethod
    def zeros_like(other: Tensor) -> Tensor:
        """
        Creates a zero-initialized tensor with the same shape and element type as the given tensor.
        """
        return Tensor.zeros(other.storage.device, other.shape, other.dtype)
