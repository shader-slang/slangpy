# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from slangpy.core.native import NativeTorchTensorDiffPair

    def diffPair(primal, grad):
        return NativeTorchTensorDiffPair(primal, grad)

except ImportError:
    pass
