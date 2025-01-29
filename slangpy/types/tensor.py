# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from slangpy.backend import Device, Buffer, ResourceUsage, TypeReflection, uint4, BufferCursor, CommandBuffer
from slangpy.core.utils import shape_to_contiguous_strides
from slangpy.reflection import SlangType, ScalarType, SlangProgramLayout
from slangpy.reflection import reflectiontypes
from slangpy.core.native import Shape
from slangpy.core.shapes import TShapeOrTuple
from slangpy.types.buffer import get_lookup_module, resolve_element_type, resolve_program_layout

from typing import Optional, Any
import numpy as np
import math

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


class Tensor:
    """
    Represents an N-D view of an underlying buffer with given shape and element type,
    and has optional gradient information attached. Element type must be differentiable.

    Strides and offset can optionally be specified and are given in terms of elements, not bytes.
    If omitted, a dense N-D grid is assumed (row-major).
    """

    def __init__(self, storage: Buffer, dtype: SlangType, shape: TShapeOrTuple,
                 strides: Optional[tuple[int, ...]] = None, offset: int = 0):

        super().__init__()

        self.storage = storage
        self.dtype = dtype
        self.shape = Shape(shape)
        self.offset = offset

        self.grad_in: Optional[Tensor] = None
        self.grad_out: Optional[Tensor] = None

        if strides is None:
            strides = shape_to_contiguous_strides(self.shape.as_tuple())

        if len(strides) != len(self.shape):
            raise ValueError("Number of strides must match number of dimensions")

        self.strides = strides

    def flatten_dtype(self) -> Tensor:
        new_dtype = innermost_type(self.dtype)
        dtype_shape = self.dtype.shape.as_tuple()
        dtype_strides = shape_to_contiguous_strides(dtype_shape)
        stride_multiplier = math.prod(dtype_shape)

        new_shape = self.shape.as_tuple() + dtype_shape
        new_strides = tuple(s * stride_multiplier for s in self.strides) + dtype_strides
        new_offset = self.offset * stride_multiplier

        return Tensor(self.storage, new_dtype, new_shape, new_strides, new_offset)

    def broadcast_to(self, shape: TShapeOrTuple) -> Tensor:
        """
        Returns a new tensor view of the same buffer with the requested shape, following standard broadcasting rules.
        """
        D = len(shape) - len(self.shape)
        if D < 0:
            raise ValueError(f"Broadcast shape must be larger than tensor shape")

        shape = Shape(shape)
        st = shape.as_tuple()

        if any(a != b and a != 1 for a, b in zip(st[D:], self.shape)):
            raise ValueError(
                f"Tensor with shape {self.shape} can't be broadcast to {shape}")

        new_strides = [0] * len(shape)
        for i, (stride, dim) in enumerate(zip(self.strides, self.shape)):
            if dim > 1:
                new_strides[D + i] = stride

        return Tensor(self.storage, self.dtype, shape, tuple(new_strides), self.offset)

    def __str__(self):
        ndarray = self.to_numpy()
        if ndarray is None:
            return f"Tensor({self.shape}, {self.dtype.name})"
        else:
            return str(ndarray)

    @property
    def grad(self) -> Tensor:
        """
        Alias for grad_out
        """
        assert self.grad_out is not None
        return self.grad_out

    @property
    def element_count(self) -> int:
        element_count = 1
        for dim in self.shape:
            element_count *= dim
        return element_count

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """
        Copies tensor data into a numpy array with the same shape and strides. This may fail if the
        element type does not have an equivalent in numpy.
        """

        numpy_dtype = _slang_to_numpy(self.dtype)
        if numpy_dtype is None:
            raise ValueError(
                f"Tensor element type {self.dtype.full_name} is not compatible with numpy")
        dtype_size = self.dtype.buffer_layout.size
        elem_size = innermost_type(self.dtype).buffer_layout.size
        dtype_shape = self.dtype.shape.as_tuple()
        dtype_strides = shape_to_contiguous_strides(dtype_shape)

        shape = self.shape.as_tuple() + dtype_shape
        strides = tuple(s * dtype_size for s in self.strides) + \
            tuple(s * elem_size for s in dtype_strides)

        data = self.storage.to_numpy().view(numpy_dtype)

        return np.lib.stride_tricks.as_strided(data, shape, strides)

    def with_grads(self, grad_in: Optional[Tensor] = None, grad_out: Optional[Tensor] = None, zero: bool = False) -> Tensor:
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
        if grad_in is None and grad_out is None:
            grad_in = Tensor.empty(
                shape=self.shape, dtype=self.dtype.derivative, device=self.storage.device)
            grad_out = grad_in

        result = Tensor(self.storage, self.dtype, self.shape, self.strides, self.offset)
        result.grad_in = grad_in
        result.grad_out = grad_out

        if zero:
            if grad_in is not None:
                grad_in.clear()
            if grad_out is not None and grad_out is not grad_in:
                grad_out.clear()
        return result

    def clear(self, command_buffer: Optional[CommandBuffer] = None):
        """
        Fill the tensor with zeros. If no command buffer is provided, a new one is created and
        immediately submitted. If a command buffer is provided the clear is simply appended to it
        but not automatically submitted.
        """
        if command_buffer:
            command_buffer.clear_resource_view(self.storage.get_uav(), uint4(0, 0, 0, 0))
        else:
            cmd = self.storage.device.create_command_buffer()
            cmd.clear_resource_view(self.storage.get_uav(), uint4(0, 0, 0, 0))
            cmd.submit()

    @property
    def slangpy_signature(self) -> str:
        return f"Tensor[{self.dtype.name},{len(self.shape)}]"

    @staticmethod
    def numpy(device: Device, ndarray: np.ndarray[Any, Any]) -> Tensor:
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

        usage = ResourceUsage.shader_resource | ResourceUsage.unordered_access
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

        usage = ResourceUsage.shader_resource | ResourceUsage.unordered_access
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
        cmd = device.create_command_buffer()
        cmd.clear_resource_view(tensor.storage.get_uav(), uint4(0, 0, 0, 0))
        cmd.submit()
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

    def cursor(self, start: Optional[int] = None, count: Optional[int] = None):
        """
        Returns a BufferCursor for the buffer, starting at the given index and with the given count
        of elements.
        """
        el_stride = self.dtype.buffer_layout.stride
        size = (count or self.element_count) * el_stride
        offset = (start or 0) * el_stride
        layout = self.dtype.buffer_layout
        return BufferCursor(layout.reflection, self.storage, size, offset)

    def uniforms(self):
        """
        Returns a dictionary of uniforms for this buffer, suitable for use with a compute kernel. These
        are useful when manually passing the buffer to a kernel, rather than going via a slangpy function.
        """
        return {
            'buffer': self.storage,
            'strides': self.strides,
            'shape': self.shape.as_tuple(),
        }
