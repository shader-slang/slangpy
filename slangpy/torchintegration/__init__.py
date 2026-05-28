# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
from slangpy.core.native import NativeTorchTensorDiffPair


def diff_pair(primal: Any, grad: Any) -> NativeTorchTensorDiffPair:
    """
    Create a differentiable tensor pair for use in backward passes.

    :param primal: The primal (value) tensor.
    :param grad: The gradient tensor.
    :return: A NativeTorchTensorDiffPair wrapping the primal and gradient tensors.
    """
    return NativeTorchTensorDiffPair(primal, grad)
