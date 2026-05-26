# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# pyright: reportUnusedImport=false

from ..native_func import Tensor, TensorDesc

NativeTensor = Tensor
NativeTensorDesc = TensorDesc

__all__ = ["Tensor", "NativeTensor", "TensorDesc", "NativeTensorDesc"]
