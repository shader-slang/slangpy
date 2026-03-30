# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Python fallback implementations for TorchBridge operations.

These functions provide equivalent functionality to the native slangpy_torch
package using pure Python/PyTorch APIs. They are used when slangpy_torch
is not installed or when fallback mode is forced for testing.

Note: The fallback path is slower than the native path but provides
identical functionality.
"""

from typing import Any, Dict, Tuple

import torch

# PyTorch scalar type codes (matching c10::ScalarType and TENSOR_BRIDGE_SCALAR_* in tensor_bridge_api.h)
_SCALAR_TYPE_MAP: Dict[torch.dtype, int] = {
    torch.uint8: 0,  # TENSOR_BRIDGE_SCALAR_UINT8
    torch.int8: 1,  # TENSOR_BRIDGE_SCALAR_INT8
    torch.int16: 2,  # TENSOR_BRIDGE_SCALAR_INT16
    torch.int32: 3,  # TENSOR_BRIDGE_SCALAR_INT32
    torch.int64: 4,  # TENSOR_BRIDGE_SCALAR_INT64
    torch.float16: 5,  # TENSOR_BRIDGE_SCALAR_FLOAT16
    torch.float32: 6,  # TENSOR_BRIDGE_SCALAR_FLOAT32
    torch.float64: 7,  # TENSOR_BRIDGE_SCALAR_FLOAT64
    torch.complex64: 9,  # TENSOR_BRIDGE_SCALAR_COMPLEX64
    torch.complex128: 10,  # TENSOR_BRIDGE_SCALAR_COMPLEX128
    torch.bool: 11,  # TENSOR_BRIDGE_SCALAR_BOOL
    torch.bfloat16: 15,  # TENSOR_BRIDGE_SCALAR_BFLOAT16
}
_complex32 = getattr(torch, "complex32", None)
if _complex32 is not None:
    _SCALAR_TYPE_MAP[_complex32] = 8  # TENSOR_BRIDGE_SCALAR_COMPLEX32


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


# ---------------------------------------------------------------------------
# __cuda_array_interface__ helper for copy_to/from_buffer
# ---------------------------------------------------------------------------


class _CudaBufferView:
    """Lightweight wrapper exposing a raw CUDA pointer via ``__cuda_array_interface__``.

    PyTorch's ``torch.as_tensor()`` can consume any object that implements the
    `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_,
    creating a zero-copy tensor view over the memory. This avoids any direct
    dependency on ``ctypes``/``cudaMemcpy`` and delegates all data movement to
    PyTorch itself.
    """

    # Mapping from torch dtype to NumPy-style typestr used by __cuda_array_interface__.
    _TYPESTR: Dict[torch.dtype, str] = {
        torch.uint8: "|u1",
        torch.int8: "|i1",
        torch.int16: "<i2",
        torch.int32: "<i4",
        torch.int64: "<i8",
        torch.float16: "<f2",
        torch.float32: "<f4",
        torch.float64: "<f8",
        torch.bfloat16: "<V2",  # no NumPy equivalent; 2-byte opaque
    }

    def __init__(
        self,
        ptr: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ):
        typestr = self._TYPESTR.get(dtype)
        if typestr is None:
            raise RuntimeError(f"Unsupported dtype for CUDA array interface: {dtype}")
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 3,
            "strides": None,  # C-contiguous
        }


def copy_to_buffer(tensor: torch.Tensor, dest_ptr: int, dest_size: int) -> bool:
    """
    Copy tensor data to a CUDA buffer.

    Handles non-contiguous tensors automatically via ``Tensor.copy_()``.
    A zero-copy view of the destination pointer is created through the
    ``__cuda_array_interface__`` protocol so that PyTorch manages all
    data movement internally.

    :param tensor: Source PyTorch CUDA tensor.
    :param dest_ptr: Destination CUDA pointer as integer.
    :param dest_size: Size in bytes of destination buffer.
    :return: True on success.
    :raises RuntimeError: If tensor is not on CUDA or buffer is too small.
    """
    if not tensor.is_cuda:
        raise RuntimeError("Tensor must be on CUDA device")

    byte_size = tensor.numel() * tensor.element_size()

    if byte_size > dest_size:
        raise RuntimeError(f"Destination buffer too small: {dest_size} < {byte_size}")

    # Create a zero-copy tensor view backed by the destination CUDA pointer.
    view = _CudaBufferView(dest_ptr, (tensor.numel(),), tensor.dtype)
    dest = torch.as_tensor(view, device=tensor.device)
    dest = dest.view(tensor.shape)
    with torch.no_grad():
        dest.copy_(tensor)
    return True


def copy_from_buffer(tensor: torch.Tensor, src_ptr: int, src_size: int) -> bool:
    """
    Copy data from a CUDA buffer to a tensor.

    Handles non-contiguous destination tensors automatically via
    ``Tensor.copy_()``. A zero-copy view of the source pointer is
    created through the ``__cuda_array_interface__`` protocol.

    :param tensor: Destination PyTorch CUDA tensor.
    :param src_ptr: Source CUDA pointer as integer.
    :param src_size: Size in bytes of source buffer.
    :return: True on success.
    :raises RuntimeError: If tensor is not on CUDA or buffer is too small.
    """
    if not tensor.is_cuda:
        raise RuntimeError("Tensor must be on CUDA device")

    byte_size = tensor.numel() * tensor.element_size()
    if byte_size > src_size:
        raise RuntimeError(f"Source buffer too small: {src_size} < {byte_size}")

    # Create a zero-copy tensor view backed by the source CUDA pointer.
    view = _CudaBufferView(src_ptr, (tensor.numel(),), tensor.dtype)
    src = torch.as_tensor(view, device=tensor.device)
    src = src.view(tensor.shape)
    with torch.no_grad():
        tensor.copy_(src)
    return True


# Reverse mapping from c10::ScalarType code to torch.dtype
_SCALAR_TYPE_TO_DTYPE: Dict[int, torch.dtype] = {v: k for k, v in _SCALAR_TYPE_MAP.items()}


def create_empty_tensor(shape: list, scalar_type: int, device_index: int = 0) -> torch.Tensor:
    """
    Create an empty contiguous CUDA tensor.

    :param shape: List of dimension sizes.
    :param scalar_type: Scalar type code (TENSOR_BRIDGE_SCALAR_* constants, e.g. 6 for float32).
    :param device_index: CUDA device index.
    :return: A new empty torch.Tensor on the specified CUDA device.
    :raises ValueError: If scalar_type is not supported.
    """
    dtype = _SCALAR_TYPE_TO_DTYPE.get(scalar_type)
    if dtype is None:
        raise ValueError(f"Unsupported scalar type code: {scalar_type}")
    return torch.empty(shape, dtype=dtype, device=f"cuda:{device_index}")


def create_zeros_like(tensor: torch.Tensor) -> torch.Tensor:
    """
    Create a zero tensor with the same shape, dtype, and device as the given tensor.

    Equivalent to torch.zeros_like(tensor).

    :param tensor: PyTorch tensor to match.
    :return: A new zero torch.Tensor with same properties.
    :raises ValueError: If object is not a PyTorch tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Object is not a PyTorch tensor")
    return torch.zeros_like(tensor)


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
