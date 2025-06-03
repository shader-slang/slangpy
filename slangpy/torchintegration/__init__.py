# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Only import things that load torch lazily here!
from .torchmodule import TorchModule


__all__ = [
    "TorchModule",
]
