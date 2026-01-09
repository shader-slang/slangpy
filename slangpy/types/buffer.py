# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from os import PathLike
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from slangpy.core.native import Shape, NativeNDBuffer, NativeNDBufferDesc
from slangpy.core.shapes import TShapeOrTuple

from slangpy import Device, MemoryType, BufferUsage, CommandEncoder
from slangpy.reflection import SlangProgramLayout, SlangType
from slangpy.reflection.lookup import resolve_program_layout, resolve_element_type, numpy_to_slang
from slangpy.types.common import load_buffer_data_from_image

import numpy as np

if TYPE_CHECKING:
    import torch


class NDBuffer(NativeNDBuffer):
    """
    An N dimensional buffer of a given slang type. The supplied type can come from a SlangType (via
    reflection), a struct read from a Module, or simply a name.

    When specifying just a type name, it is advisable to also supply the program_layout for the
    module in question (see Module.layout), as this ensures type information is looked up from
    the right place.
    """

    def __init__(
        self,
        device: Device,
        dtype: Any,
        element_count: Optional[int] = None,
        shape: Optional[TShapeOrTuple] = None,
        usage: BufferUsage = BufferUsage.shader_resource | BufferUsage.unordered_access,
        memory_type: MemoryType = MemoryType.device_local,
        program_layout: Optional[SlangProgramLayout] = None,
    ):
        if element_count is None and shape is None:
            raise ValueError("Either element_count or shape must be provided")
        if element_count is not None and shape is not None:
            raise ValueError("Only one of element_count or shape can be provided")

        # Slang program layout of module that defines the element type for this buffer.
        program_layout = resolve_program_layout(device, dtype, program_layout)

        # Slang element type.
        dtype = resolve_element_type(program_layout, dtype)

        if element_count is None:
            if shape is None:
                raise ValueError("Either element_count or shape must be provided")
            element_count = 1
            for dim in shape:
                element_count *= dim
            shape = Shape(shape)
        elif shape is None:
            if element_count is None:
                raise ValueError("Either element_count or shape must be provided")
            shape = Shape(element_count)
        else:
            raise ValueError("element_count or shape must be provided")

        desc = NativeNDBufferDesc()
        desc.usage = usage
        desc.memory_type = memory_type
        desc.shape = shape
        desc.strides = shape.calc_contiguous_strides()
        desc.dtype = dtype
        desc.element_layout = dtype.buffer_layout.reflection

        super().__init__(device, desc)

        # Tell typing the dtype is a valid slang type
        self.dtype: "SlangType"

    @property
    def is_writable(self):
        """
        Returns True if this buffer is writable from the GPU, i.e. if it has unordered access resource usage.
        """
        return (self.usage & BufferUsage.unordered_access) != 0

    def broadcast_to(self, shape: TShapeOrTuple):
        """
        Returns a new view of the buffer with the requested shape, following standard broadcasting rules.
        """
        return super().broadcast_to(Shape(shape))

    def view(self, shape: TShapeOrTuple, strides: TShapeOrTuple = Shape(), offset: int = 0):
        """
        Returns a new view of the tensor with the requested shape, strides and offset
        The offset is in elements (not bytes) and is specified relative to the current offset
        """
        return super().view(Shape(shape), Shape(strides), offset)

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """
        Copies buffer data into a numpy array with the same shape and strides. If the element type
        of the buffer is representable in numpy (e.g. floats, ints, arrays/vectors thereof), the
        ndarray will have a matching dtype. If the element type can't be represented in numpy (e.g. structs),
        the ndarray will be an array over the bytes of the buffer elements

        Examples:
        NDBuffer of dtype float3 with shape (4, 5)
            -> ndarray of dtype np.float32 with shape (4, 5, 3)
        NDBuffer of dtype struct Foo {...} with shape (5, )
            -> ndarray of dtype np.uint8 with shape (5, sizeof(Foo))
        """
        return cast(np.ndarray[Any, Any], super().to_numpy())

    def to_torch(self) -> "torch.Tensor":
        """
        Returns a view of the buffer data as a torch tensor with the same shape and strides.
        See to_numpy for notes on dtype conversion
        """
        return cast("torch.Tensor", super().to_torch())

    def clear(self, command_encoder: Optional[CommandEncoder] = None):
        """
        Fill the ndbuffer with zeros. If no command buffer is provided, a new one is created and
        immediately submitted. If a command buffer is provided the clear is simply appended to it
        but not automatically submitted.
        """
        super().clear(command_encoder)

    @staticmethod
    def from_numpy(
        device: Device,
        ndarray: np.ndarray[Any, Any],
        usage: BufferUsage = BufferUsage.shader_resource | BufferUsage.unordered_access,
        memory_type: MemoryType = MemoryType.device_local,
        program_layout: Optional[SlangProgramLayout] = None,
        shape: Optional[TShapeOrTuple] = None,
        dtype: Optional[Any] = None,
    ) -> "NDBuffer":
        """
        Creates a new NDBuffer with the same contents, shape and strides as the given numpy array.
        """

        if dtype is None:
            dtype = numpy_to_slang(ndarray.dtype, device, program_layout)
            if dtype is None:
                raise ValueError(f"Unsupported numpy dtype {ndarray.dtype}")
            if not ndarray.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "Currently NDBuffers can only be directly constructed from C-contiguous numpy arrays"
                )
        if shape is None:
            shape = ndarray.shape

        res = NDBuffer(
            device,
            dtype=dtype,
            shape=shape,
            usage=usage,
            memory_type=memory_type,
        )
        res.copy_from_numpy(ndarray)
        return res

    @staticmethod
    def empty(
        device: Device,
        shape: TShapeOrTuple,
        dtype: Any,
        usage: BufferUsage = BufferUsage.shader_resource | BufferUsage.unordered_access,
        memory_type: MemoryType = MemoryType.device_local,
        program_layout: Optional[SlangProgramLayout] = None,
    ) -> "NDBuffer":
        """
        Creates an NDBuffer with the requested shape and element type without attempting to initialize the data.
        """
        return NDBuffer(
            device,
            dtype=dtype,
            shape=shape,
            usage=usage,
            memory_type=memory_type,
            program_layout=program_layout,
        )

    @staticmethod
    def zeros(
        device: Device,
        shape: TShapeOrTuple,
        dtype: Any,
        usage: BufferUsage = BufferUsage.shader_resource | BufferUsage.unordered_access,
        memory_type: MemoryType = MemoryType.device_local,
        program_layout: Optional[SlangProgramLayout] = None,
    ) -> "NDBuffer":
        """
        Creates a zero-initialized nbuffer with the requested shape and element type.
        """
        buffer = NDBuffer.empty(device, shape, dtype, usage, memory_type, program_layout)
        buffer.clear()
        return buffer

    @staticmethod
    def empty_like(other: "NDBuffer") -> "NDBuffer":
        """
        Creates a new tensor with the same shape and element type as the given tensor, without initializing the data.
        """
        return NDBuffer.empty(
            other.device, other.shape, other.dtype, other.usage, other.memory_type
        )

    @staticmethod
    def zeros_like(other: "NDBuffer") -> "NDBuffer":
        """
        Creates a zero-initialized ndbuffer with the same shape and element type as the given ndbuffer.
        """
        return NDBuffer.zeros(
            other.device, other.shape, other.dtype, other.usage, other.memory_type
        )

    @staticmethod
    def load_from_image(
        device: Device,
        path: Union[str, PathLike[str]],
        flip_y: bool = False,
        linearize: bool = False,
        scale: float = 1.0,
        offset: float = 0.0,
        grayscale: bool = False,
    ) -> "NDBuffer":
        """
        Helper to load an image from a file and convert it to a floating point tensor.
        """

        # Load bitmap + convert to numpy array
        data = load_buffer_data_from_image(path, flip_y, linearize, scale, offset, grayscale)

        # Create buffer with appropriate dtype based on number of channels.
        if len(data.shape) == 2 or data.shape[2] == 1:
            dtype = "float"
        elif data.shape[2] == 2:
            dtype = "float2"
        elif data.shape[2] == 3:
            dtype = "float3"
        elif data.shape[2] == 4:
            dtype = "float4"
        else:
            raise ValueError(f"Unsupported number of channels: {data.shape[2]}")
        buffer = NDBuffer.empty(device, data.shape[:2], dtype)
        buffer.copy_from_numpy(data)
        return buffer
