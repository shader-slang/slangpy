# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
PyTorch autograd integration for SlangPy.

This module provides automatic differentiation support for SlangPy kernels
when used with PyTorch tensors that have requires_grad=True.

The implementation tracks tensors during dispatch using the access patterns
from the bound variable runtime (read/write/readwrite) to determine which
tensors are inputs vs outputs for autograd purposes.

NOTE: This module is currently a work-in-progress. The TensorRef-based
implementation has been removed in favor of a simpler approach that tracks
tensors directly during dispatch.
"""

from typing import TYPE_CHECKING, Any, Optional, Tuple
from dataclasses import dataclass
import torch

from slangpy.core.native import AccessType, NativeCallRuntimeOptions
from slangpy import Device, DeviceType

if TYPE_CHECKING:
    from slangpy.core.function import FunctionNode
    from slangpy.core.calldata import CallData


def check_cuda_enabled(device: Device) -> None:
    """
    Check that CUDA interop is enabled for the device.

    Args:
        device: The SGL device to check.

    Raises:
        RuntimeError: If CUDA interop is not enabled.
    """
    if not device.supports_cuda_interop and device.info.type != DeviceType.cuda:
        raise RuntimeError(
            "Cuda interop must be enabled for torch support. "
            "Create SGL device with Device(..., enable_cuda_interop=True)"
        )


@dataclass
class TrackedTensor:
    """
    Information about a tensor tracked for autograd purposes.

    Attributes:
        tensor: The PyTorch tensor.
        access: The access type (read, write, readwrite) for this tensor.
        arg_name: The argument name/path this tensor came from (for debugging).
    """

    tensor: torch.Tensor
    access: AccessType
    arg_name: str


class TorchAutoGradHook(torch.autograd.Function):
    """
    PyTorch autograd function that wraps SlangPy kernel dispatch.

    This allows gradients to flow through SlangPy kernel calls when
    tensors have requires_grad=True.

    The forward pass:
    1. Extracts all tensors from the arguments
    2. Runs the SlangPy kernel
    3. Returns output tensors connected to the autograd graph

    The backward pass:
    1. Receives gradients for output tensors
    2. Calls the .bwds() version of the SlangPy function
    3. Returns gradients for input tensors
    """

    @staticmethod
    def forward(
        ctx: Any,
        options: Tuple[
            "CallData",
            "CallData",
            NativeCallRuntimeOptions,
            list[Any],
            dict[str, Any],
            list[torch.Tensor],
            list[torch.Tensor],
        ],
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Store data for backward pass
        ctx.options = options
        forwards_cd = options[0]
        rt_options = options[2]
        args = options[3]
        kwargs = options[4]
        inputs = options[5]
        outputs = options[6]

        # Run the kernel
        result = forwards_cd.call(rt_options, *args, **kwargs)

        # Save all tensors for backward
        ctx.save_for_backward(*inputs, *outputs)

        return tuple(outputs)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Optional[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass - compute gradients by calling .bwds().

        Args:
            ctx: PyTorch autograd context with saved state.
            *grad_outputs: Gradients for each output tensor.

        Returns:
            Tuple of gradients for each input, with None for non-tensor args.
        """
        # TODO: Implement backward pass
        # This requires:
        # 1. Creating gradient tensors for inputs
        # 2. Calling function.bwds() with the gradients
        # 3. Returning gradients in the correct order

        raise NotImplementedError(
            "Backward pass for autograd is not yet implemented. "
            "This will call function.bwds() with gradient tensors."
        )
