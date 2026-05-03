# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Torch tensor detection utilities.

This module provides functions to detect torch.Tensor objects in function arguments
without requiring TensorRef wrappers.
"""

from typing import Any
from slangpy.core.native import NativeTorchTensorDiffPair

# Cached torch module reference (None = not checked yet, False = not available)
_torch_module: Any = None


def detect_torch_tensors(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[bool, bool]:
    """
    Scan args/kwargs for torch.Tensor objects.

    This function recursively scans through args and kwargs to find any
    torch.Tensor objects. It's used during call data creation to determine
    if torch integration is needed.

    Args:
        args: Positional arguments (already unpacked from IThis wrappers)
        kwargs: Keyword arguments (already unpacked from IThis wrappers)

    Returns:
        A tuple of:
        - has_torch_tensors: True if any torch tensors were found
        - requires_autograd: True if any tensor has requires_grad=True
    """

    # Check for torch just once and cache the module
    global _torch_module
    if _torch_module is None:
        try:
            import torch

            _torch_module = torch
        except ImportError:
            _torch_module = False
    if _torch_module is False:
        return False, False

    # Local reference to torch module for use in nested function
    torch = _torch_module

    has_torch = False
    requires_autograd = False

    def scan(obj: Any) -> None:
        nonlocal has_torch, requires_autograd

        if isinstance(obj, torch.Tensor):
            has_torch = True
            if obj.requires_grad:
                requires_autograd = True
        elif isinstance(obj, NativeTorchTensorDiffPair):
            if obj.primal is not None or obj.grad is not None:
                has_torch = True
        elif isinstance(obj, dict):
            for v in obj.values():
                scan(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                scan(v)
        # Note: We don't recurse into arbitrary objects - only standard containers

    for arg in args:
        scan(arg)
    for v in kwargs.values():
        scan(v)

    return has_torch, requires_autograd
