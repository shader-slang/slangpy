# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Torch tensor detection utilities.

This module provides functions to detect torch.Tensor objects in function arguments
without requiring TensorRef wrappers.
"""

from typing import Any

TORCH_AVAILABLE = None


def detect_torch_tensors(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[bool, bool, list[Any]]:
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
        - tensors: List of all detected torch.Tensor objects
    """

    # Check for torch just once
    global TORCH_AVAILABLE
    if TORCH_AVAILABLE is None:
        try:
            import torch

            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False
    if not TORCH_AVAILABLE:
        return False, False, []

    tensors: list[Any] = []
    requires_autograd = False

    def scan(obj: Any) -> None:
        nonlocal requires_autograd

        if isinstance(obj, torch.Tensor):
            tensors.append(obj)
            if obj.requires_grad:
                requires_autograd = True
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

    return len(tensors) > 0, requires_autograd, tensors
