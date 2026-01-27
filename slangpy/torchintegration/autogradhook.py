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
import torch

from slangpy.core.native import NativeCallRuntimeOptions, NativeTorchTensorDiffPair

if TYPE_CHECKING:
    from slangpy.core.function import FunctionNode
    from slangpy.core.calldata import CallData


class TorchAutoGradHook(torch.autograd.Function):
    """
    PyTorch autograd function that wraps SlangPy kernel dispatch.

    This allows gradients to flow through SlangPy kernel calls when
    tensors have requires_grad=True.

    The forward pass:
    1. Receives args/kwargs with torch tensors already wrapped in NativeTorchTensorDiffPair
    2. Runs the SlangPy kernel
    3. Returns output tensors connected to the autograd graph

    The backward pass:
    1. Receives gradients for output tensors
    2. Restores tensor references and populates gradient fields
    3. Generates backwards CallData from function.bwds with real gradient tensors
    4. Calls the backwards kernel
    5. Returns gradients for input tensors
    """

    @staticmethod
    def forward(
        ctx: Any,
        options: Tuple[
            "FunctionNode",
            "CallData",
            NativeCallRuntimeOptions,
            list[Any],
            dict[str, Any],
            list[Any],  # NativeTorchTensorDiffPair list
        ],
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass - run the SlangPy kernel and track tensors for backward.

        :param ctx: PyTorch autograd context for storing state.
        :param options: Tuple containing (function, forwards_cd, rt_options, args, kwargs, pairs).
        :param tensors: Input tensors tracked by autograd (primals of input pairs).
        :return: Tuple of output tensors.
        """
        # Unpack options
        function = options[0]
        forwards_cd = options[1]
        rt_options = options[2]
        args = options[3]
        kwargs = options[4]
        pairs = options[5]

        # Store function for backward pass (needed to generate backwards CallData)
        ctx.function = function
        ctx.forwards_cd = forwards_cd
        ctx.rt_options = rt_options

        # Collect inputs and outputs from pairs for save_for_backward
        inputs: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        for pair in pairs:
            if pair.is_input:
                inputs.append(pair.primal)
            else:
                outputs.append(pair.primal)

        # Run the forward kernel - args/kwargs already contain NativeTorchTensorDiffPair objects
        result = forwards_cd.call(rt_options, *args, **kwargs)

        # Clear tensor references from pairs to avoid keeping them alive,
        # but keep the pairs themselves for backward pass
        for pair in pairs:
            pair.clear_tensors()
        ctx.args = args
        ctx.kwargs = kwargs
        ctx.pairs = pairs

        # Save all tensors for backward (inputs first, then outputs)
        ctx.save_for_backward(*inputs)

        # Return any output tensors plus the result if present
        res = tuple(outputs)
        if result is not None:
            assert isinstance(result, torch.Tensor)
            if not "_result" in kwargs:
                pair = NativeTorchTensorDiffPair(None, None, len(pairs), False)
                kwargs["_result"] = pair
                ctx.pairs.append(pair)
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
        # Retrieve saved input tensors
        saved_tensors = ctx.saved_tensors
        saved_inputs = list(saved_tensors)

        # Restore tensor references to pairs and populate gradients
        input_idx = 0
        grad_output_idx = 0
        input_grads: list[Optional[torch.Tensor]] = []
        for pair in ctx.pairs:
            if pair.is_input:
                # Restore primal tensor
                pair.primal = saved_inputs[input_idx]
                # Create gradient tensor for this input
                if pair.primal.requires_grad:
                    pair.grad = torch.zeros_like(pair.primal)
                    input_grads.append(pair.grad)
                else:
                    pair.grad = None
                    input_grads.append(None)
                input_idx += 1
            else:
                # For outputs, we don't need the primal value in backward pass,
                # just need shape/dtype info. Use a meta tensor (zero memory).
                assert grad_output_idx < len(grad_outputs)
                grad_out = grad_outputs[grad_output_idx]
                if grad_out is not None:
                    pair.primal = None
                    pair.grad = grad_out
                else:
                    pair.primal = None
                    pair.grad = None
                grad_output_idx += 1

        # By fixing up pairs, we will have implicitly reconstructed the args/kwargs structure
        # so can just call the backwards immediately.
        args = ctx.args
        kwargs = ctx.kwargs
        function = ctx.function
        function.bwds(*args, **kwargs)

        # 6. Return gradients in correct order: None for options tuple, then input grads
        # The first argument to forward() is the options tuple, so we return None for it
        return (None,) + tuple(input_grads)
