# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
from slangpy.core.native import NativeTorchTensorDiffPair


def diff_pair(primal: Any, grad: Any, is_input: bool = True) -> NativeTorchTensorDiffPair:
    """
    Create a differentiable tensor pair for use in backward passes.

    :param primal: The primal (value) tensor.
    :param grad: The gradient tensor.
    :param is_input: True if this is an input (kernel writes gradients),
        False if this is an output (kernel reads upstream gradients).
    :return: A NativeTorchTensorDiffPair wrapping the primal and gradient tensors.
    """
    return NativeTorchTensorDiffPair(primal, grad, -1, is_input)
