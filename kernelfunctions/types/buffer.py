from typing import Any, Optional

from sgl import MemoryType, TypeReflection

from kernelfunctions.backend import Device, ResourceUsage, TypeLayoutReflection

from kernelfunctions.core import BaseType, Shape
from kernelfunctions.core.reflection import SlangProgramLayout, SlangType
from kernelfunctions.shapes import TShapeOrTuple
from kernelfunctions.struct import Struct
from kernelfunctions.typeregistry import get_or_create_type

import numpy.typing as npt

global_lookup_modules: dict[Device, SlangProgramLayout] = {}


def get_lookup_module(device: Device) -> SlangProgramLayout:
    if device not in global_lookup_modules:
        dummy_module = device.load_module_from_source(
            "slangpy_layout", 'import "slangpy";')
        global_lookup_modules[device] = SlangProgramLayout(dummy_module.layout)
    return global_lookup_modules[device]


def resolve_program_layout(device: Device, element_type: Any, program_layout: Optional[SlangProgramLayout]) -> SlangProgramLayout:
    if program_layout is None:
        if isinstance(element_type, SlangType):
            program_layout = element_type.program
        elif isinstance(element_type, BaseType):
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
    elif isinstance(element_type, BaseType):
        if element_type.slang_type.program == program_layout:
            element_type = element_type.slang_type
        else:
            element_type = program_layout.find_type_by_name(
                element_type.slang_type.full_name)
    else:
        bt = get_or_create_type(program_layout, element_type)
        element_type = bt.slang_type
    if element_type is None:
        raise ValueError("Element type could not be resolved")
    return element_type


class NDBuffer:
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

        self.program_layout = resolve_program_layout(device, element_type, program_layout)
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

        self.usage = usage

        self.slangpy_signature = f"[{self.element_type.full_name},{len(self.shape)},{self.is_writable}]"

        strides = []
        total = 1
        for dim in reversed(self.shape):
            strides.append(total)
            total *= dim
        self.strides = tuple(reversed(strides))

        self.element_size = self.element_type.buffer_layout.size
        self.element_stride = self.element_type.buffer_layout.stride
        self.device = device

        self.buffer = device.create_buffer(
            element_count=self.element_count,
            struct_size=self.element_size,
            usage=self.usage,
            memory_type=memory_type
        )

    @property
    def is_writable(self):
        return (self.usage & ResourceUsage.unordered_access) != 0

    def to_numpy(self):
        return self.buffer.to_numpy()

    def from_numpy(self, data: npt.ArrayLike):
        self.buffer.from_numpy(data)


class NDDifferentiableBuffer(NDBuffer):
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
            grad_type = self.element_type

        self.grad_type = resolve_element_type(self.program_layout, element_type)

        self.requires_grad = requires_grad

        if grad_usage is not None:
            usage = grad_usage
        if grad_memory_type is not None:
            memory_type = grad_memory_type

        if self.requires_grad:
            self.grad = NDDifferentiableBuffer(
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

        self.grad_usage = grad_usage if grad_usage is not None else self.usage

    @property
    def is_differentiable(self):
        return self.requires_grad

    @property
    def is_writable(self):
        return (self.usage & ResourceUsage.unordered_access) != 0

    def primal_to_numpy(self):
        return self.to_numpy()

    def primal_from_numpy(self, data: npt.ArrayLike):
        self.from_numpy(data)

    def grad_to_numpy(self):
        assert self.grad is not None
        return self.grad.to_numpy()

    def grad_from_numpy(self, data: npt.ArrayLike):
        assert self.grad is not None
        self.grad.from_numpy(data)
