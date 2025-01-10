# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

import numpy.typing as npt

from slangpy.core.native import Shape
from slangpy.core.shapes import TShapeOrTuple
from slangpy.core.struct import Struct

from slangpy.backend import (BufferCursor, DataType, Device, MemoryType,
                             ResourceUsage, TypeLayoutReflection,
                             TypeReflection)
from slangpy.bindings.marshall import Marshall
from slangpy.bindings.typeregistry import get_or_create_type
from slangpy.reflection import ScalarType, SlangProgramLayout, SlangType

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


def _on_device_close(device: Device):
    del global_lookup_modules[device]


def get_lookup_module(device: Device) -> SlangProgramLayout:
    if device not in global_lookup_modules:
        dummy_module = device.load_module_from_source(
            "slangpy_layout", 'import "slangpy";')
        global_lookup_modules[device] = SlangProgramLayout(dummy_module.layout)
        device.register_device_close_callback(_on_device_close)

    return global_lookup_modules[device]


def resolve_program_layout(device: Device, element_type: Any, program_layout: Optional[SlangProgramLayout]) -> SlangProgramLayout:
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
            element_type = program_layout.find_type_by_name(element_type.name)
    elif isinstance(element_type, TypeReflection):
        element_type = program_layout.find_type(element_type)
    elif isinstance(element_type, TypeLayoutReflection):
        element_type = program_layout.find_type(element_type.type)
    elif isinstance(element_type, Marshall):
        if element_type.slang_type.program == program_layout:
            element_type = element_type.slang_type
        else:
            element_type = program_layout.find_type_by_name(
                element_type.slang_type.full_name)
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


class NDBuffer:
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
        element_type: Any,
        element_count: Optional[int] = None,
        shape: Optional[TShapeOrTuple] = None,
        usage: ResourceUsage = ResourceUsage.shader_resource
        | ResourceUsage.unordered_access,
        memory_type: MemoryType = MemoryType.device_local,
        program_layout: Optional[SlangProgramLayout] = None
    ):
        super().__init__()

        if element_count is None and shape is None:
            raise ValueError("Either element_count or shape must be provided")
        if element_count is not None and shape is not None:
            raise ValueError("Only one of element_count or shape can be provided")

        #: Slang program layout of module that defines the element type for this buffer.
        self.program_layout = resolve_program_layout(device, element_type, program_layout)

        #: Slang element type.
        self.element_type = resolve_element_type(self.program_layout, element_type)

        if element_count is None:
            if shape is None:
                raise ValueError("Either element_count or shape must be provided")
            element_count = 1
            for dim in shape:
                element_count *= dim
            self.element_count = element_count
            self.shape = Shape(shape)
        elif shape is None:
            if element_count is None:
                raise ValueError("Either element_count or shape must be provided")
            self.element_count = element_count
            self.shape = Shape(element_count)

        #: Buffer resource usage.
        self.usage = usage

        #: Slangpy type signature.
        self.slangpy_signature = f"[{self.element_type.full_name},{len(self.shape)},{self.is_writable}]"

        strides = []
        total = 1
        for dim in reversed(self.shape):
            strides.append(total)
            total *= dim
        self.strides = tuple(reversed(strides))

        #: Element size in bytes.
        self.element_size = self.element_type.buffer_layout.size

        #: Element stride in bytes.
        self.element_stride = self.element_type.buffer_layout.stride

        #: SGL device.
        self.device = device

        #: Internal structured buffer.
        self.buffer = device.create_buffer(
            element_count=self.element_count,
            struct_size=self.element_size,
            usage=self.usage,
            memory_type=memory_type
        )

    @property
    def is_writable(self):
        """
        Returns True if this buffer is writable from the GPU, i.e. if it has unordered access resource usage.
        """
        return (self.usage & ResourceUsage.unordered_access) != 0

    def to_numpy(self):
        """
        Returns the buffer as a numpy array.
        """
        return self.buffer.to_numpy()

    def from_numpy(self, data: npt.ArrayLike):
        """
        Sets the buffer from a numpy array.
        """
        self.buffer.from_numpy(data)

    def to_torch(self, override_type: Optional[DataType] = None):
        """
        Returns the buffer as a torch tensor.
        """
        if isinstance(self.element_type, ScalarType):
            return self.buffer.to_torch(type=SLANG_TO_CUDA_TYPES[self.element_type.slang_scalar_type], shape=self.shape.as_tuple(), strides=self.strides)
        else:
            raise ValueError("Only scalar types can be converted to torch tensors")

    def cursor(self, start: Optional[int] = None, count: Optional[int] = None):
        """
        Returns a BufferCursor for the buffer, starting at the given index and with the given count
        of elements.
        """
        size = (count or self.element_count) * self.element_stride
        offset = (start or 0) * self.element_stride
        layout = self.element_type.buffer_layout
        return BufferCursor(layout.reflection, self.buffer, size, offset)

    def uniforms(self):
        """
        Returns a dictionary of uniforms for this buffer, suitable for use with a compute kernel. These
        are useful when manually passing the buffer to a kernel, rather than going via a slangpy function.
        """
        return {
            'buffer': self.buffer,
            'strides': self.strides,
            'shape': self.shape.as_tuple(),
        }


class DeprecatedNDDifferentiableBuffer(NDBuffer):
    """
    Deprecated: Use slangpy.Tensor instead.

    An N dimensional buffer of a given slang type, with optional additional buffer of gradients. 
    The supplied type can come from a SlangType (via reflection), a struct read from a Module, 
    or simply a name. If unspecified, the type of the gradient is assumed to match that of the 
    primal.

    When specifying just a type name, it is advisable to also supply the program_layout for the
    module in question (see Module.layout), as this ensures type information is looked up from
    the right place.
    """

    def __init__(
        self,
        device: Device,
        element_type: Any,
        element_count: Optional[int] = None,
        shape: Optional[TShapeOrTuple] = None,
        usage: ResourceUsage = ResourceUsage.shader_resource
        | ResourceUsage.unordered_access,
        memory_type: MemoryType = MemoryType.device_local,
        requires_grad: bool = False,
        grad_type: Any = None,
        grad_usage: Optional[ResourceUsage] = None,
        grad_memory_type: Optional[MemoryType] = None,
        program_layout: Optional[SlangProgramLayout] = None
    ):
        super().__init__(device, element_type, element_count, shape, usage, memory_type, program_layout)

        if grad_type is None:
            grad_type = self.element_type.derivative

        #: Slang element type for the gradient.
        self.grad_type = resolve_element_type(self.program_layout, element_type)

        #: Whether gradient buffer is required.
        self.requires_grad = requires_grad

        if grad_usage is not None:
            usage = grad_usage
        if grad_memory_type is not None:
            memory_type = grad_memory_type

        if self.requires_grad:
            #: Gradient buffer.
            self.grad = DeprecatedNDDifferentiableBuffer(
                device=device,
                element_type=grad_type,
                element_count=element_count,
                shape=shape,
                usage=usage,
                memory_type=memory_type,
                requires_grad=False,
                grad_type=None,
                grad_usage=None,
                grad_memory_type=None,
                program_layout=self.program_layout)
            self.slangpy_signature += self.grad.slangpy_signature
        else:
            self.grad = None
            self.slangpy_signature += "[]"

        #: Gradient resource usage.
        self.grad_usage = grad_usage if grad_usage is not None else self.usage

    @property
    def is_differentiable(self):
        """
        Returns True if this buffer is differentiable, i.e. if it has a gradient.
        """
        return self.requires_grad

    @property
    def is_writable(self):
        """
        Returns True if this buffer is writable from the GPU, i.e. if it has unordered access resource usage.
        """
        return (self.usage & ResourceUsage.unordered_access) != 0

    def primal_to_numpy(self):
        """
        Returns the primal buffer as a numpy array (alias for to_numpy).
        """
        return self.to_numpy()

    def primal_from_numpy(self, data: npt.ArrayLike):
        """
        Sets the primal buffer from a numpy array (alias for from_numpy).
        """
        self.from_numpy(data)

    def primal_to_torch(self):
        """
        Returns the primal buffer as a torch tensor (alias for to_torch).
        """
        return self.to_torch()

    def grad_to_numpy(self):
        """
        Returns the gradient buffer as a numpy array.
        """
        assert self.grad is not None
        return self.grad.to_numpy()

    def grad_from_numpy(self, data: npt.ArrayLike):
        """
        Sets the gradient buffer from a numpy array.
        """
        assert self.grad is not None
        self.grad.from_numpy(data)

    def grad_to_torch(self):
        """
        Returns the gradient buffer as a torch tensor.
        """
        assert self.grad is not None
        return self.grad.to_torch()

    def get_grad(self):
        """
        Returns the gradient buffer, raising exception if not valid.
        """
        assert self.grad is not None
        return self.grad
