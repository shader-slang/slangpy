# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
PyTorch autograd integration for SlangPy.

This module provides automatic differentiation support for SlangPy kernels
when used with PyTorch tensors that have requires_grad=True.

Forward and backward logic is implemented natively in C++
(NativeCallData::autograd_forward / autograd_backward). This Python class
is a thin wrapper required by the torch.autograd.Function API.
"""

from typing import TYPE_CHECKING, Any, Optional, Tuple
import torch

from slangpy.core.native import NativeCallRuntimeOptions

if TYPE_CHECKING:
    from slangpy.core.function import FunctionNode
    from slangpy.core.calldata import CallData


class TorchAutoGradHook(torch.autograd.Function):
    """
    PyTorch autograd function that wraps SlangPy kernel dispatch.

    Forward and backward are thin wrappers that delegate to the native
    C++ methods NativeCallData::autograd_forward and autograd_backward.
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
        Forward pass - delegates to NativeCallData::autograd_forward.

        :param ctx: PyTorch autograd context for storing state.
        :param options: Tuple containing (function, forwards_cd, rt_options, args, kwargs, pairs).
        :param tensors: Input tensors tracked by autograd (primals of input pairs).
        :return: Tuple of output tensors.
        """
        function, forwards_cd, rt_options, args, kwargs, pairs = options

        ctx.function = function
        ctx.forwards_cd = forwards_cd

        # Native C++ handles: kernel dispatch, pair bookkeeping, output collection
        input_tensors, output_tensors, result, pairs = forwards_cd.autograd_forward(
            rt_options, args, kwargs, pairs
        )

        ctx.args = args
        ctx.kwargs = kwargs
        ctx.pairs = pairs
        ctx.save_for_backward(*input_tensors)

        # output_tensors already includes the result tensor (if any) —
        # autograd_forward appends it when creating the _result pair.
        return tuple(output_tensors)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Optional[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass - delegates to NativeCallData::autograd_backward.

        :param ctx: PyTorch autograd context with saved state.
        :param grad_outputs: Gradients for each output tensor.
        :return: Tuple of gradients for each input, with None for the options arg.
        """
        # Native C++ handles: tensor restoration, grad creation, bwds dispatch
        input_grads = ctx.forwards_cd.autograd_backward(
            ctx.function,
            ctx.pairs,
            ctx.args,
            ctx.kwargs,
            list(ctx.saved_tensors),
            grad_outputs,
        )

        return (None,) + tuple(input_grads)
