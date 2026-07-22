# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# pyright: reportUnusedImport=false

# Compatibility shim for the old location of Tensor and TensorDesc.
# These should now be imported from slangpy.Tensor
from ..native_func import Tensor, TensorDesc

__all__ = ["Tensor", "TensorDesc"]
