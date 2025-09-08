# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import math
from os import PathLike
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from slangpy.core.native import Shape, NativeNDBuffer, NativeNDBufferDesc
from slangpy.core.shapes import TShapeOrTuple
from slangpy.core.struct import Struct

from slangpy import (
    DataType,
    Device,
    MemoryType,
    BufferUsage,
    TypeLayoutReflection,
    TypeReflection,
    CommandEncoder,
    Bitmap,
    DataStruct,
)
from slangpy.bindings.marshall import Marshall
from slangpy.bindings.typeregistry import get_or_create_type
from slangpy.reflection import ScalarType, SlangProgramLayout, SlangType, reflectiontypes

import numpy as np

if TYPE_CHECKING:
    import torch

global_lookup_modules: dict[Device, SlangProgramLayout] = {}

SLANG_TO_CUDA_TYPES = {
    TypeReflection.ScalarType.float16: DataType.float16,
    TypeReflection.ScalarType.float32: DataType.float32,
    TypeReflection.ScalarType.float64: DataType.float64,
    TypeReflection.ScalarType.int8: DataType.int8,
    TypeReflection.ScalarType.int16: DataType.int16,
    TypeReflection.ScalarType.int32: DataType.int32,
    TypeReflection.ScalarType.int64: DataType.int64,
    TypeReflection.ScalarType.uint8: DataType.uint8,
    TypeReflection.ScalarType.uint16: DataType.uint16,
    TypeReflection.ScalarType.uint32: DataType.uint32,
    TypeReflection.ScalarType.uint64: DataType.uint64,
    TypeReflection.ScalarType.bool: DataType.bool,
}

ST = TypeReflection.ScalarType
_numpy_to_sgl = {
    "int8": ST.int8,
    "int16": ST.int16,
    "int32": ST.int32,
    "int64": ST.int64,
    "uint8": ST.uint8,
    "uint16": ST.uint16,
    "uint32": ST.uint32,
    "uint64": ST.uint64,
    "float16": ST.float16,
    "float32": ST.float32,
    "float64": ST.float64,
}
_sgl_to_numpy = {y: x for x, y in _numpy_to_sgl.items()}


def _on_device_close(device: Device):
    del global_lookup_modules[device]


def _load_lookup_module(device: Device):
    dummy_module = device.load_module_from_source("slangpy_layout", 'import "slangpy";')
    global_lookup_modules[device] = SlangProgramLayout(dummy_module.layout)


def get_lookup_module(device: Device) -> SlangProgramLayout:
    if device not in global_lookup_modules:
        _load_lookup_module(device)
        device.register_device_close_callback(_on_device_close)
        device.register_shader_hot_reload_callback(lambda _: _load_lookup_module(device))

    return global_lookup_modules[device]


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


def _numpy_to_slang(
    np_dtype: np.dtype[Any], device: Device, program_layout: Optional[SlangProgramLayout]
) -> Optional[SlangType]:
    name = np_dtype.base.name
    if name not in _numpy_to_sgl:
        return None
    slang_dtype = reflectiontypes.scalar_names[_numpy_to_sgl[name]]
    if np_dtype.ndim > 0:
        for dim in reversed(np_dtype.shape):
            slang_dtype += f"[{dim}]"

    if program_layout is None:
        program_layout = get_lookup_module(device)
    return program_layout.find_type_by_name(slang_dtype)


def resolve_program_layout(
    device: Device, element_type: Any, program_layout: Optional[SlangProgramLayout]
) -> SlangProgramLayout:
    if program_layout is None:
        if isinstance(element_type, SlangType):
            program_layout = element_type.program
        elif isinstance(element_type, Marshall):
            program_layout = element_type.slang_type.program
        elif isinstance(element_type, Struct):
            program_layout = element_type.module.layout
        else:
            program_layout = get_lookup_module(device)
    return program_layout


def resolve_element_type(program_layout: SlangProgramLayout, element_type: Any) -> SlangType:
    if isinstance(element_type, SlangType):
        pass
    elif isinstance(element_type, str):
        element_type = program_layout.find_type_by_name(element_type)
    elif isinstance(element_type, Struct):
        if element_type.module.layout == program_layout:
            element_type = element_type.struct
        else:
            element_type = program_layout.find_type_by_name(element_type.full_name)
    elif isinstance(element_type, TypeReflection):
        element_type = program_layout.find_type_by_name(element_type.full_name)
    elif isinstance(element_type, TypeLayoutReflection):
        element_type = program_layout.find_type_by_name(element_type.type.full_name)
    elif isinstance(element_type, Marshall):
        if element_type.slang_type.program == program_layout:
            element_type = element_type.slang_type
        else:
            element_type = program_layout.find_type_by_name(element_type.slang_type.full_name)
    # elif element_type == float:
    #    element_type = program_layout.scalar_type(TypeReflection.ScalarType.float32)
    # elif element_type == int:
    #    element_type = program_layout.scalar_type(TypeReflection.ScalarType.int32)
    else:
        bt = get_or_create_type(program_layout, element_type)
        element_type = bt.slang_type
    if element_type is None:
        raise ValueError("Element type could not be resolved")
    return element_type


def load_buffer_data_from_image(
    path: Union[str, PathLike[str]],
    flip_y: bool = False,
    linearize: bool = False,
    scale: float = 1.0,
    offset: float = 0.0,
    greyscale: bool = False,
) -> np.ndarray[Any, Any]:
    """
    Helper to load an image from a file and convert it to a floating point tensor.
    """ ""

    # Load bitmap + convert to numpy array
    bitmap = Bitmap(path)

    # Select target pixel format based on channel count and greyscale flag.
    pix_fmt = bitmap.pixel_format
    if greyscale:
        pix_fmt = Bitmap.PixelFormat.r
    else:
        if bitmap.channel_count == 1:
            pix_fmt = Bitmap.PixelFormat.r
        elif bitmap.channel_count == 2:
            pix_fmt = Bitmap.PixelFormat.rg
        elif bitmap.channel_count == 3:
            pix_fmt = Bitmap.PixelFormat.rgb
        elif bitmap.channel_count == 4:
            pix_fmt = Bitmap.PixelFormat.rgba

    # Select whether to de-gamma the bitmap based on linearization flag.
    if linearize:
        srgb_gamma = False
    else:
        srgb_gamma = bitmap.srgb_gamma

    # Perform conversion to the desired pixel format.
    bitmap = bitmap.convert(pix_fmt, DataStruct.Type.float32, srgb_gamma)

    # Convert bitmap to numpy array.
    data: np.ndarray[Any, Any] = np.array(bitmap, copy=False)

    # Validate array shape.
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(f"Bitmap data must be 2 or 3 dimensional, got {data.ndim} dimensions")
    if data.ndim == 3:
        if data.shape[2] not in [1, 2, 3, 4]:
            raise ValueError(
                f"Bitmap data must have 1, 2, 3 or 4 channels, got {data.shape[2]} channels"
            )
    if data.dtype != np.float32:
        raise ValueError(f"Bitmap data must be float32, got {data.dtype}")

    # Flip if requested
    if flip_y:
        data = np.flipud(data)

    # Apply scale and offset if requested.
    if scale != 1.0 or offset != 0.0:
        data = data * scale + offset

    return data


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
    ) -> "NDBuffer":
        """
        Creates a new NDBuffer with the same contents, shape and strides as the given numpy array.
        """

        dtype = _numpy_to_slang(ndarray.dtype, device, program_layout)
        if dtype is None:
            raise ValueError(f"Unsupported numpy dtype {ndarray.dtype}")
        if not ndarray.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "Currently NDBuffers can only be directly constructed from C-contiguous numpy arrays"
            )

        res = NDBuffer(
            device,
            dtype=dtype,
            shape=ndarray.shape,
            usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
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
