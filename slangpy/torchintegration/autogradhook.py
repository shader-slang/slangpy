# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
PyTorch autograd integration for SlangPy.

This module provides automatic differentiation support for SlangPy kernels
when used with PyTorch tensors that have requires_grad=True.

The implementation tracks tensors during dispatch using the access patterns
from the bound variable runtime (read/write/readwrite) to determine which
tensors are inputs vs outputs for autograd purposes.

The backwards CallData is generated lazily during the backward pass when actual
gradient tensors are available, rather than being pre-generated at forward time.
This allows SlangPy's type analysis to work naturally with the real gradient tensors.
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

    :param device: The SGL device to check.
    :raises RuntimeError: If CUDA interop is not enabled.
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


def _build_backwards_args(
    fwd_args: tuple[Any, ...],
    fwd_kwargs: dict[str, Any],
    fwd_runtime: Any,
    inputs: list[torch.Tensor],
    input_grads: list[Optional[torch.Tensor]],
    outputs: list[torch.Tensor],
    output_grads: list[Optional[torch.Tensor]],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Build arguments for the backwards kernel call.

    For the backwards pass, we need to provide:
    - Primal values for inputs (for computing gradients)
    - Gradient tensors for inputs (to write computed gradients into)
    - Gradient tensors for outputs (the incoming gradients from upstream)

    This function walks the argument structure and replaces torch.Tensor values
    appropriately for the backwards call.

    :param fwd_args: Original positional arguments from forward pass.
    :param fwd_kwargs: Original keyword arguments from forward pass.
    :param fwd_runtime: The BoundCallRuntime from the forwards CallData.
    :param inputs: List of input tensors (those that were read).
    :param input_grads: List of gradient tensors for inputs (or None if no grad needed).
    :param outputs: List of output tensors (those that were written).
    :param output_grads: List of gradient tensors for outputs (incoming gradients).
    :return: Tuple of (bwds_args, bwds_kwargs) ready for the backwards kernel.
    """
    # Track which input/output we're processing
    input_idx = [0]
    output_idx = [0]

    def process_value(
        value: Any,
        binding: Any,
    ) -> Any:
        """Recursively process a value, replacing tensors as needed."""
        if isinstance(value, dict):
            return {
                k: process_value(
                    v, binding.children[k] if binding and hasattr(binding, "children") else None
                )
                for k, v in value.items()
            }
        elif isinstance(value, torch.Tensor):
            # Determine if this tensor was an input or output based on access pattern
            from slangpy.reflection import ITensorType, TensorAccess

            is_input = False
            if binding is not None:
                if isinstance(binding.vector_type, ITensorType):
                    is_input = binding.vector_type.access == TensorAccess.read
                else:
                    is_input = binding.access[0] == AccessType.read

            if is_input:
                # Input tensor: return the primal (backwards needs it for computation)
                # The gradient will be written by the kernel
                idx = input_idx[0]
                input_idx[0] += 1
                # For backwards, we just pass the primal tensor
                # The gradient tensor is handled separately via d_out marshalling
                return value
            else:
                # Output tensor: the gradient is what we need to pass
                idx = output_idx[0]
                output_idx[0] += 1
                if idx < len(output_grads) and output_grads[idx] is not None:
                    return output_grads[idx]
                return value
        else:
            return value

    # Process args
    bwds_args = tuple(
        process_value(arg, fwd_runtime.args[i] if i < len(fwd_runtime.args) else None)
        for i, arg in enumerate(fwd_args)
    )

    # Process kwargs (excluding _result which is handled separately)
    bwds_kwargs = {
        k: process_value(v, fwd_runtime.kwargs.get(k))
        for k, v in fwd_kwargs.items()
        if k != "_result"
    }

    return bwds_args, bwds_kwargs


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
    2. Generates backwards CallData from function.bwds with real gradient tensors
    3. Calls the backwards kernel
    4. Returns gradients for input tensors
    """

    @staticmethod
    def forward(
        ctx: Any,
        options: Tuple[
            "FunctionNode",
            "CallData",
            NativeCallRuntimeOptions,
            tuple[Any, ...],
            dict[str, Any],
            list[torch.Tensor],
            list[torch.Tensor],
        ],
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass - run the SlangPy kernel and track tensors for backward.

        :param ctx: PyTorch autograd context for storing state.
        :param options: Tuple containing (function, forwards_cd, rt_options, args, kwargs, inputs, outputs).
        :param tensors: Input tensors tracked by autograd.
        :return: Tuple of output tensors.
        """
        # Unpack options
        function = options[0]
        forwards_cd = options[1]
        rt_options = options[2]
        args = options[3]
        kwargs = options[4]
        inputs = options[5]
        outputs = options[6]

        # Store data for backward pass - function is needed to generate backwards CallData
        ctx.function = function
        ctx.forwards_cd = forwards_cd
        ctx.rt_options = rt_options
        ctx.args = args
        ctx.kwargs = kwargs
        ctx.num_inputs = len(inputs)
        ctx.num_outputs = len(outputs)

        # Run the forward kernel
        result = forwards_cd.call(rt_options, *args, **kwargs)

        # Save all tensors for backward (inputs first, then outputs)
        ctx.save_for_backward(*inputs, *outputs)

        # Return any output tensors plus the result if present
        res = tuple(outputs)
        if result is not None:
            res += (result,)
        return res

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Optional[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass - compute gradients by calling function.bwds().

        The backwards CallData is generated here with actual gradient tensors,
        allowing SlangPy's type analysis to work naturally.

        :param ctx: PyTorch autograd context with saved state.
        :param grad_outputs: Gradients for each output tensor.
        :return: Tuple of gradients for each input, with None for the options arg.
        """
        # 1. Retrieve saved tensors and split into inputs/outputs
        saved_tensors = ctx.saved_tensors
        inputs = list(saved_tensors[: ctx.num_inputs])
        outputs = list(saved_tensors[ctx.num_inputs :])

        # 2. Create gradient tensors for inputs (zeros_like for those that need grads)
        input_grads: list[Optional[torch.Tensor]] = []
        for inp in inputs:
            if inp.requires_grad:
                input_grads.append(torch.zeros_like(inp))
            else:
                input_grads.append(None)

        # 3. Build backwards args/kwargs by replacing tensors with (primal, grad) pairs
        # The backwards kernel expects DiffTensorPair-like structures where we have
        # both the primal value and a gradient tensor to write to.
        bwds_args, bwds_kwargs = _build_backwards_args(
            ctx.args,
            ctx.kwargs,
            ctx.forwards_cd.runtime,
            inputs,
            input_grads,
            outputs,
            list(grad_outputs),
        )

        # 4. If the function has a return value, the output gradient becomes _result
        # The gradient of the return value is the last grad_output (if result was returned)
        if len(grad_outputs) > len(outputs):
            # There was a return value - its gradient is the last grad_output
            result_grad = grad_outputs[-1]
            if result_grad is not None:
                bwds_kwargs["_result"] = result_grad

        # 5. Generate backwards CallData and call the kernel
        bwds_func = ctx.function.bwds
        bwds_call_data = bwds_func.generate_call_data(bwds_args, bwds_kwargs)
        bwds_call_data.call(ctx.rt_options, *bwds_args, **bwds_kwargs)

        # 6. Return gradients in correct order: None for options tuple, then input grads
        # The first argument to forward() is the options tuple, so we return None for it
        return (None,) + tuple(input_grads)
