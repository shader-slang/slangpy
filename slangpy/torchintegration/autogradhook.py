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

from typing import TYPE_CHECKING, Any, Optional
from dataclasses import dataclass
import torch

from slangpy.core.native import AccessType
from slangpy import Device, DeviceType

if TYPE_CHECKING:
    from slangpy.core.function import FunctionNode


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
        function: "FunctionNode",
        call_data: Any,  # NativeCallData
        options: Any,  # NativeCallRuntimeOptions
        tracked_tensors: list[TrackedTensor],
        unpacked_args: tuple[Any, ...],
        unpacked_kwargs: dict[str, Any],
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass - run the SlangPy kernel.

        Args:
            ctx: PyTorch autograd context for saving state.
            function: The FunctionNode to call.
            call_data: The NativeCallData for the kernel.
            options: Runtime options for the call.
            tracked_tensors: List of TrackedTensor with access info.
            unpacked_args: Unpacked positional arguments.
            unpacked_kwargs: Unpacked keyword arguments.
            *tensors: All tensors flattened for autograd tracking.

        Returns:
            Tuple of output tensors (those with write or readwrite access).
        """
        # Store data for backward pass
        ctx.function = function
        ctx.call_data = call_data
        ctx.tracked_tensors = tracked_tensors
        ctx.unpacked_args = unpacked_args
        ctx.unpacked_kwargs = unpacked_kwargs

        # Run the kernel
        result = call_data.call(options, *unpacked_args, **unpacked_kwargs)

        # Identify output tensors (written or read-write)
        output_tensors: list[torch.Tensor] = []
        inout_tensors: list[torch.Tensor] = []

        for tt in tracked_tensors:
            if tt.access in (AccessType.write, AccessType.readwrite):
                output_tensors.append(tt.tensor)
            if tt.access == AccessType.readwrite:
                inout_tensors.append(tt.tensor)

        # If function returns a tensor, add it to outputs
        if isinstance(result, torch.Tensor):
            if not any(t is result for t in output_tensors):
                output_tensors.append(result)

        # Mark in-out tensors as dirty
        ctx.mark_dirty(*inout_tensors)

        # Save all tensors for backward
        all_tensors = [tt.tensor for tt in tracked_tensors]
        ctx.save_for_backward(*all_tensors)

        return tuple(output_tensors)

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
