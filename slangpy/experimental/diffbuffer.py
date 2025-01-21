# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

import numpy.typing as npt

from slangpy.core.shapes import TShapeOrTuple

from slangpy.backend import (Device, MemoryType,
                             ResourceUsage)
from slangpy.reflection import SlangProgramLayout
from slangpy.types.buffer import NDBuffer, resolve_element_type, resolve_program_layout


class NDDifferentiableBuffer(NDBuffer):
    """
    WIP: Use slangpy.Tensor instead.

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
            grad_type = self.dtype.derivative

        program_layout = resolve_program_layout(device, grad_type, program_layout)

        #: Slang element type for the gradient.
        self.grad_type = resolve_element_type(program_layout, element_type)

        #: Whether gradient buffer is required.
        self.requires_grad = requires_grad

        if grad_usage is not None:
            usage = grad_usage
        if grad_memory_type is not None:
            memory_type = grad_memory_type

        if self.requires_grad:
            #: Gradient buffer.
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
                program_layout=program_layout)
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
