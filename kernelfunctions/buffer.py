from typing import Optional, Type, Union
import sgl

from kernelfunctions.shapes import TConcreteShape
from .typemappings import TSGLVector, TPythonScalar

ALL_SUPPORTED_TYPES = Union[TSGLVector, TPythonScalar, sgl.TypeLayoutReflection]


def _calc_element_type_size(element_type: type) -> int:
    if element_type in (sgl.int1, sgl.uint1, sgl.float1, sgl.bool1, int, float, bool):
        return 4
    elif element_type in (sgl.int2, sgl.uint2, sgl.float2, sgl.bool2):
        return 8
    elif element_type in (sgl.int3, sgl.uint3, sgl.float3, sgl.bool3):
        return 12
    elif element_type in (sgl.int4, sgl.uint4, sgl.float4, sgl.bool4):
        return 16
    raise ValueError(f"Unsupported type: {element_type}")


class StructuredBuffer:
    def __init__(
        self,
        device: sgl.Device,
        element_type: Type[ALL_SUPPORTED_TYPES],
        element_count: Optional[int] = None,
        shape: Optional[TConcreteShape] = None,
        usage: sgl.ResourceUsage = sgl.ResourceUsage.shader_resource
        | sgl.ResourceUsage.unordered_access,
        requires_grad: bool = False,
        grad_type: Optional[Type[ALL_SUPPORTED_TYPES]] = None,
        grad_usage: Optional[sgl.ResourceUsage] = None,
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

        self.requires_grad = requires_grad
        self.grad_type = grad_type if grad_type is not None else self.element_type
        self.grad_usage = grad_usage if grad_usage is not None else self.usage

        self.element_size = _calc_element_type_size(self.element_type)
        self.grad_element_size = _calc_element_type_size(self.grad_type)

        self.buffer = device.create_buffer(
            element_count=self.element_count,
            struct_size=self.element_size,
            usage=self.usage,
        )

        if self.requires_grad:
            self.grad_buffer = device.create_buffer(
                element_count=self.element_count,
                struct_size=self.grad_element_size,
                usage=self.grad_usage,
            )
        else:
            self.grad_buffer = None
