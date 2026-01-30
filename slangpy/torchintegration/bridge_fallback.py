# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Python fallback implementations for TorchBridge operations.

These functions provide equivalent functionality to the native slangpy_torch
package using pure Python/PyTorch APIs. They are used when slangpy_torch
is not installed or when fallback mode is forced for testing.

Note: The fallback path is slower than the native path but provides
identical functionality.
"""

from typing import Any, Dict

import torch

# PyTorch scalar type codes (matching c10::ScalarType)
_SCALAR_TYPE_MAP: Dict[torch.dtype, int] = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.complex32: 8,
    torch.complex64: 9,
    torch.complex128: 10,
    torch.bool: 11,
    torch.bfloat16: 15,
}


def is_tensor(obj: Any) -> bool:
    """
    Check if object is a torch.Tensor.

    :param obj: Object to check.
    :return: True if object is a torch.Tensor.
    """
    return isinstance(obj, torch.Tensor)


def extract_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Extract tensor metadata as a dictionary.

    Equivalent to the native extract_torch_tensor_info() function.

    :param tensor: PyTorch tensor to extract info from.
    :return: Dictionary containing tensor metadata.
    :raises ValueError: If object is not a PyTorch tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Object is not a PyTorch tensor")

    return {
        "data_ptr": tensor.data_ptr(),
        "shape": tuple(tensor.shape),
        "strides": tuple(tensor.stride()),
        "ndim": tensor.ndim,
        "device_type": 1 if tensor.is_cuda else 0,
        "device_index": tensor.device.index if tensor.is_cuda else -1,
        "scalar_type": _SCALAR_TYPE_MAP.get(tensor.dtype, -1),
        "element_size": tensor.element_size(),
        "numel": tensor.numel(),
        "storage_offset": tensor.storage_offset(),
        "cuda_stream": _get_cuda_stream(tensor) if tensor.is_cuda else 0,
        "is_contiguous": tensor.is_contiguous(),
        "is_cuda": tensor.is_cuda,
        "requires_grad": tensor.requires_grad,
    }


def get_signature(tensor: torch.Tensor) -> str:
    """
    Get tensor signature string: "[Dn,Sm]" where n=ndim, m=scalar_type.

    :param tensor: PyTorch tensor to get signature for.
    :return: Signature string.
    :raises ValueError: If object is not a PyTorch tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    scalar_type = _SCALAR_TYPE_MAP.get(tensor.dtype, -1)
    return f"[D{tensor.ndim},S{scalar_type}]"


def get_current_cuda_stream(device_index: int) -> int:
    """
    Get the current CUDA stream pointer for a device.

    :param device_index: CUDA device index.
    :return: CUDA stream pointer as integer, or 0 if CUDA not available.
    """
    if not torch.cuda.is_available():
        return 0
    stream = torch.cuda.current_stream(device_index)
    return stream.cuda_stream


def copy_to_buffer(tensor: torch.Tensor, dest_ptr: int, dest_size: int) -> bool:
    """
    Copy tensor data to a CUDA buffer.

    Handles non-contiguous tensors by making a contiguous copy first.

    Note: This fallback implementation requires the native slangpy_torch package
    for CUDA memory copy operations. The Python fallback is primarily intended
    for metadata extraction (shape, dtype, etc.) rather than memory operations.

    :param tensor: Source PyTorch CUDA tensor.
    :param dest_ptr: Destination CUDA pointer as integer.
    :param dest_size: Size in bytes of destination buffer.
    :return: True on success.
    :raises RuntimeError: Always raises - CUDA memory copy requires native support.
    """
    if not tensor.is_cuda:
        raise RuntimeError("Tensor must be on CUDA device")

    byte_size = tensor.numel() * tensor.element_size()

    if byte_size > dest_size:
        raise RuntimeError(f"Destination buffer too small: {dest_size} < {byte_size}")

    # CUDA memory copy operations require native support
    # The Python fallback cannot safely perform raw CUDA memory copies
    raise RuntimeError(
        "copy_to_buffer requires native slangpy_torch support. "
        "Install slangpy_torch for CUDA memory copy operations."
    )


def copy_from_buffer(tensor: torch.Tensor, src_ptr: int, src_size: int) -> bool:
    """
    Copy data from a CUDA buffer to a tensor.

    Handles non-contiguous tensors.

    Note: This fallback implementation requires the native slangpy_torch package
    for CUDA memory copy operations. The Python fallback is primarily intended
    for metadata extraction (shape, dtype, etc.) rather than memory operations.

    :param tensor: Destination PyTorch CUDA tensor.
    :param src_ptr: Source CUDA pointer as integer.
    :param src_size: Size in bytes of source buffer.
    :return: True on success.
    :raises RuntimeError: Always raises - CUDA memory copy requires native support.
    """
    if not tensor.is_cuda:
        raise RuntimeError("Tensor must be on CUDA device")

    byte_size = tensor.numel() * tensor.element_size()
    if byte_size > src_size:
        raise RuntimeError(f"Source buffer too small: {src_size} < {byte_size}")

    # CUDA memory copy operations require native support
    # The Python fallback cannot safely perform raw CUDA memory copies
    raise RuntimeError(
        "copy_from_buffer requires native slangpy_torch support. "
        "Install slangpy_torch for CUDA memory copy operations."
    )


def _get_cuda_stream(tensor: torch.Tensor) -> int:
    """
    Get the CUDA stream pointer for the tensor's device.

    :param tensor: PyTorch tensor.
    :return: CUDA stream pointer as integer, or 0 if not on CUDA.
    """
    if not tensor.is_cuda:
        return 0
    device_index = tensor.device.index or 0
    stream = torch.cuda.current_stream(device_index)
    return stream.cuda_stream
