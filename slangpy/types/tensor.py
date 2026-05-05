# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tensor is now a native C++ class exposed via nanobind.

This module re-exports Tensor and NativeTensor from the native extension, and
adds the ``load_from_image`` and deprecated ``numpy`` class methods that rely
on heavy Python-side helpers (Bitmap / numpy image processing).
"""
from __future__ import annotations
from os import PathLike
import warnings

from slangpy import Device
from slangpy.core.native import NativeTensor
from slangpy.types.common import load_buffer_data_from_image

from typing import Any, Union

import numpy as np

# The canonical Tensor type is the C++ NativeTensor, exposed as "Tensor" in the binding.
# Both names are exported for backward compatibility.
Tensor = NativeTensor


def _tensor_load_from_image(
    device: Device,
    path: Union[str, PathLike[str]],
    flip_y: bool = False,
    linearize: bool = False,
    scale: float = 1.0,
    offset: float = 0.0,
    grayscale: bool = False,
) -> NativeTensor:
    """
    Helper to load an image from a file and convert it to a floating point tensor.
    """
    data = load_buffer_data_from_image(path, flip_y, linearize, scale, offset, grayscale)

    if len(data.shape) == 2 or data.shape[2] == 1:
        dtype = "float"
    elif data.shape[2] == 2:
        dtype = "float2"
    elif data.shape[2] == 3:
        dtype = "float3"
    elif data.shape[2] == 4:
        dtype = "float4"
    else:
        raise ValueError(f"Unsupported number of channels: {data.shape[2]}")
    tensor = Tensor.empty(device, data.shape[:2], dtype)
    tensor.copy_from_numpy(data)
    return tensor


def _tensor_numpy(device: Device, ndarray: np.ndarray[Any, Any]) -> NativeTensor:
    warnings.warn(
        "Tensor.numpy is deprecated. Use Tensor.from_numpy instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Tensor.from_numpy(device, ndarray)


# Attach as class methods so users can call Tensor.load_from_image(...) etc.
Tensor.load_from_image = staticmethod(_tensor_load_from_image)  # type: ignore[attr-defined]
Tensor.numpy = staticmethod(_tensor_numpy)  # type: ignore[attr-defined]
