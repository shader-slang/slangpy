# SPDX-License-Identifier: Apache-2.0
# pyright: reportUnusedImport=false

TORCH_ENABLED = False

try:
    import torch  # @IgnoreException
    TORCH_ENABLED = True
except ImportError:
    pass

if TORCH_ENABLED:
    from .torchfunction import TorchFunction
    from .torchmodule import TorchModule
    from .torchstruct import TorchStruct
