from typing import Any, Optional

from kernelfunctions.backend import Device, ResourceUsage, TypeLayoutReflection

from kernelfunctions.shapes import TConcreteShape
from kernelfunctions.typeregistry import get_or_create_type


class NDBuffer:
    def __init__(
        self,
        device: Device,
        element_type: Any,
        element_count: Optional[int] = None,
        shape: Optional[TConcreteShape] = None,
        usage: ResourceUsage = ResourceUsage.shader_resource
        | ResourceUsage.unordered_access,
    ):
        super().__init__()

        if element_count is None and shape is None:
            raise ValueError("Either element_count or shape must be provided")
        if element_count is not None and shape is not None:
            raise ValueError("Only one of element_count or shape can be provided")

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
            self.shape = (element_count,)

        self.element_type = get_or_create_type(element_type)
        self.usage = usage

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
            self.element_size = self.element_type.byte_size()
            self.element_stride = self.element_size

        self.buffer = device.create_buffer(
            element_count=self.element_count,
            struct_size=self.element_size,
            usage=self.usage,
        )

    @property
    def is_writable(self):
        return (self.usage & ResourceUsage.unordered_access) != 0


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
    ):
        super().__init__(device, element_type, element_count, shape, usage)

        if grad_type is None:
            grad_type = element_type

        self.requires_grad = requires_grad
        self.grad_type = get_or_create_type(grad_type)
        self.grad_usage = grad_usage if grad_usage is not None else self.usage

        if isinstance(grad_type, TypeLayoutReflection):
            self.grad_size = grad_type.size
            self.grad_stride = grad_type.stride
        else:
            self.grad_size = self.grad_type.byte_size()
            self.grad_stride = self.grad_size

        if self.requires_grad:
            self.grad_buffer = device.create_buffer(
                element_count=self.element_count,
                struct_size=self.grad_size,
                usage=self.grad_usage,
            )
        else:
            self.grad_buffer = None

    @property
    def is_differentiable(self):
        return self.requires_grad

    @property
    def is_writable(self):
        return (self.usage & ResourceUsage.unordered_access) != 0
