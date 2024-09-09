from typing import Optional, Type, Union
from kernelfunctions.backend import Device, ResourceUsage, TypeLayoutReflection

from kernelfunctions.shapes import TConcreteShape
from ..typemappings import TSGLVector, TPythonScalar, calc_element_type_size

ALL_SUPPORTED_TYPES = Union[Type[TSGLVector],
                            Type[TPythonScalar], TypeLayoutReflection]


class NDBuffer:
    def __init__(
        self,
        device: Device,
        element_type: ALL_SUPPORTED_TYPES,
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

        self.element_type = element_type
        self.usage = usage

        strides = []
        total = 1
        for dim in reversed(self.shape):
            strides.append(total)
            total *= dim
        self.strides = tuple(reversed(strides))

        self.element_size = calc_element_type_size(self.element_type)

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
        element_type: ALL_SUPPORTED_TYPES,
        element_count: Optional[int] = None,
        shape: Optional[TConcreteShape] = None,
        usage: ResourceUsage = ResourceUsage.shader_resource
        | ResourceUsage.unordered_access,
        requires_grad: bool = False,
        grad_type: Optional[ALL_SUPPORTED_TYPES] = None,
        grad_usage: Optional[ResourceUsage] = None,
    ):
        super().__init__(device, element_type, element_count, shape, usage)

        self.requires_grad = requires_grad
        self.grad_type = grad_type if grad_type is not None else self.element_type
        self.grad_usage = grad_usage if grad_usage is not None else self.usage
        self.grad_element_size = calc_element_type_size(self.grad_type)

        if self.requires_grad:
            self.grad_buffer = device.create_buffer(
                element_count=self.element_count,
                struct_size=self.grad_element_size,
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
