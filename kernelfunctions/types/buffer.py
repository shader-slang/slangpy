from typing import Any, Optional

from sgl import TypeReflection

from kernelfunctions.backend import Device, ResourceUsage, TypeLayoutReflection, SlangModule

from kernelfunctions.core.basetype import BaseType
from kernelfunctions.shapes import TConcreteShape
from kernelfunctions.struct import Struct
from kernelfunctions.typeregistry import get_or_create_type
from kernelfunctions.utils import find_type_layout_for_buffer

import numpy.typing as npt


class NDBuffer:
    def __init__(
        self,
        device: Device,
        element_type: Any,
        element_count: Optional[int] = None,
        shape: Optional[TConcreteShape] = None,
        usage: ResourceUsage = ResourceUsage.shader_resource
        | ResourceUsage.unordered_access,
        slang_module: Optional[SlangModule] = None,
    ):
        super().__init__()

        if element_count is None and shape is None:
            raise ValueError("Either element_count or shape must be provided")
        if element_count is not None and shape is not None:
            raise ValueError("Only one of element_count or shape can be provided")

        if isinstance(element_type, str):
            if slang_module is None:
                raise ValueError(
                    "slang_module must be provided to resolve string based element types")
            element_type = find_type_layout_for_buffer(slang_module.layout, element_type)
        elif isinstance(element_type, Struct):
            element_type = find_type_layout_for_buffer(
                element_type.device_module.layout, element_type.name)

        if element_count is None:
            if shape is None:
                raise ValueError("Either element_count or shape must be provided")
            element_count = 1
            for dim in shape:
                element_count *= dim
            self.element_count = element_count
            self.shape = shape
        elif shape is None:
            if element_count is None:
                raise ValueError("Either element_count or shape must be provided")
            self.element_count = element_count
            self.shape: tuple[int, ...] = (element_count,)

        self.element_type = get_or_create_type(element_type)
        self.usage = usage

        self._cached_signature = f"[{self.element_type.name},{len(self.shape)},{self.is_writable}]"

        strides = []
        total = 1
        for dim in reversed(self.shape):
            strides.append(total)
            total *= dim
        self.strides = tuple(reversed(strides))

        if isinstance(element_type, TypeLayoutReflection):
            self.element_size = element_type.size
            self.element_stride = element_type.stride
        else:
            self.element_size = self.element_type.get_byte_size()
            self.element_stride = self.element_size

        self.buffer = device.create_buffer(
            element_count=self.element_count,
            struct_size=self.element_size,
            usage=self.usage,
        )

    @property
    def is_writable(self):
        return (self.usage & ResourceUsage.unordered_access) != 0

    @property
    def slangpy_signature(self) -> str:
        return self._cached_signature

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
        shape: Optional[TConcreteShape] = None,
        usage: ResourceUsage = ResourceUsage.shader_resource
        | ResourceUsage.unordered_access,
        requires_grad: bool = False,
        grad_type: Any = None,
        grad_usage: Optional[ResourceUsage] = None,
        slang_module: Optional[SlangModule] = None,
    ):
        super().__init__(device, element_type, element_count, shape, usage, slang_module)

        if grad_type is None:
            if isinstance(element_type, BaseType):
                grad_type = element_type.derivative
            elif isinstance(element_type, Struct):
                grad_type = find_type_layout_for_buffer(
                    element_type.device_module.layout, element_type.name+".Differential")
            elif isinstance(element_type, str):
                if slang_module is None:
                    raise ValueError(
                        "slang_module must be provided to resolve string based element types")
                grad_type = find_type_layout_for_buffer(
                    slang_module.layout, element_type+".Differential")
            elif isinstance(element_type, (TypeLayoutReflection, TypeReflection)):
                if slang_module is not None:
                    grad_type = element_type.name+".Differential"
            if grad_type is None:
                grad_type = element_type

        self.requires_grad = requires_grad

        if self.requires_grad:
            self.grad = NDDifferentiableBuffer(
                device=device,
                element_type=grad_type,
                element_count=element_count,
                shape=shape,
                usage=usage,
                requires_grad=False,
                grad_type=None,
                grad_usage=None,
                slang_module=slang_module)
        else:
            self.grad = None

        self.grad_type = get_or_create_type(grad_type)
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
