# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    import torch
    from slangpy.core.native import NativeTorchTensorDiffPair

    class DiffPair(NativeTorchTensorDiffPair):
        """A pair of torch tensors (primal + gradient) for differentiable SlangPy kernel calls."""

        def __init__(self, primal: torch.Tensor, grad: torch.Tensor):
            super().__init__(primal, grad)

except ImportError:
    pass
