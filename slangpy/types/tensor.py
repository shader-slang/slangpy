# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from slangpy.backend import Device, Buffer, ResourceUsage, TypeReflection, uint4
from slangpy.core.utils import shape_to_contiguous_strides
from slangpy.reflection import SlangType, ScalarType, SlangProgramLayout
from slangpy.reflection import reflectiontypes
from .buffer import get_lookup_module, resolve_element_type, resolve_program_layout

from typing import Optional, cast, Any
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

    def __init__(self, storage: Buffer, dtype: SlangType, shape: tuple[int, ...],
                 strides: Optional[tuple[int, ...]] = None, offset: int = 0):

        super().__init__()

        self.storage = storage
        self.dtype = dtype
        self.shape = shape
        self.offset = offset

        self.grad_in: Optional[Tensor] = None
        self.grad_out: Optional[Tensor] = None

        if strides is None:
            strides = shape_to_contiguous_strides(self.shape)

        if len(strides) != len(self.shape):
            raise ValueError("Number of strides must match number of dimensions")

        self.strides = strides

    def flatten_dtype(self) -> Tensor:
        new_dtype = innermost_type(self.dtype)
        dtype_shape = self.dtype.shape.as_tuple()
        dtype_strides = shape_to_contiguous_strides(dtype_shape)
        stride_multiplier = math.prod(dtype_shape)

        new_shape = self.shape + dtype_shape
        new_strides = tuple(s * stride_multiplier for s in self.strides) + dtype_strides
        new_offset = self.offset * stride_multiplier

        return Tensor(self.storage, new_dtype, new_shape, new_strides, new_offset)

    def broadcast_to(self, shape: tuple[int, ...]) -> Tensor:
        """
        Returns a new tensor view of the same buffer with the requested shape, following standard broadcasting rules.
        """
        D = len(shape) - len(self.shape)
        if D < 0:
            raise ValueError(f"Broadcast shape must be larger than tensor shape")
        if any(a != b and a != 1 for a, b in zip(shape[D:], self.shape)):
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
        dtype_shape = tuple(cast(int, x) for x in self.dtype.shape)
        dtype_strides = shape_to_contiguous_strides(dtype_shape)

        shape = self.shape + dtype_shape
        strides = tuple(s * dtype_size for s in self.strides) + \
            tuple(s * elem_size for s in dtype_strides)

        data = self.storage.to_numpy().view(numpy_dtype)

        return np.lib.stride_tricks.as_strided(data, shape, strides)

    def with_grads(self, grad_in: Optional[Tensor] = None, grad_out: Optional[Tensor] = None) -> Tensor:
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
            grad_in = Tensor.zeros(
                shape=self.shape, dtype=self.dtype.derivative, device=self.storage.device)
            grad_out = grad_in

        result = Tensor(self.storage, self.dtype, self.shape, self.strides, self.offset)
        result.grad_in = grad_in
        result.grad_out = grad_out
        return result

    @property
    def slangpy_signature(self) -> str:
        return f"Tensor[{self.dtype.name},{len(self.shape)}]"

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

        usage = ResourceUsage.shader_resource | ResourceUsage.unordered_access
        buffer = device.create_buffer(ndarray.nbytes, usage=usage, data=flattened)

        return Tensor(buffer, dtype, tuple(ndarray.shape), strides)

    @staticmethod
    def empty(device: Device, shape: tuple[int, ...], dtype: Any, program_layout: Optional[SlangProgramLayout] = None) -> Tensor:
        """
        Creates a tensor with the requested shape and element type without attempting to initialize the data.
        """

        # If dtype supplied is not a SlangType, resolve it using the same mechanism as NDBuffer
        if not isinstance(dtype, SlangType):
            program_layout = resolve_program_layout(device, dtype, program_layout)
            dtype = resolve_element_type(program_layout, dtype)

        usage = ResourceUsage.shader_resource | ResourceUsage.unordered_access
        buffer = device.create_buffer(
            dtype.buffer_layout.size * math.prod(shape), usage=usage)

        return Tensor(buffer, dtype, shape)

    @staticmethod
    def zeros(device: Device, shape: tuple[int, ...], dtype: Any) -> Tensor:
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
