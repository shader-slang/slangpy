# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy import Bitmap, DataStruct
import numpy as np
from os import PathLike
from typing import Any, Union


def load_buffer_data_from_image(
    path: Union[str, PathLike[str]],
    flip_y: bool = False,
    linearize: bool = False,
    scale: float = 1.0,
    offset: float = 0.0,
    greyscale: bool = False,
) -> np.ndarray[Any, Any]:
    """
    Helper to load an image from a file and convert it to a floating point tensor.
    """ ""

    # Load bitmap + convert to numpy array
    bitmap = Bitmap(path)

    # Select target pixel format based on channel count and greyscale flag.
    pix_fmt = bitmap.pixel_format
    if greyscale:
        pix_fmt = Bitmap.PixelFormat.r
    else:
        if bitmap.channel_count == 1:
            pix_fmt = Bitmap.PixelFormat.r
        elif bitmap.channel_count == 2:
            pix_fmt = Bitmap.PixelFormat.rg
        elif bitmap.channel_count == 3:
            pix_fmt = Bitmap.PixelFormat.rgb
        elif bitmap.channel_count == 4:
            pix_fmt = Bitmap.PixelFormat.rgba

    # Select whether to de-gamma the bitmap based on linearization flag.
    if linearize:
        srgb_gamma = False
    else:
        srgb_gamma = bitmap.srgb_gamma

    # Perform conversion to the desired pixel format.
    bitmap = bitmap.convert(pix_fmt, DataStruct.Type.float32, srgb_gamma)

    # Convert bitmap to numpy array.
    data: np.ndarray[Any, Any] = np.array(bitmap, copy=False)

    # Validate array shape.
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(f"Bitmap data must be 2 or 3 dimensional, got {data.ndim} dimensions")
    if data.ndim == 3:
        if data.shape[2] not in [1, 2, 3, 4]:
            raise ValueError(
                f"Bitmap data must have 1, 2, 3 or 4 channels, got {data.shape[2]} channels"
            )
    if data.dtype != np.float32:
        raise ValueError(f"Bitmap data must be float32, got {data.dtype}")

    # Flip if requested
    if flip_y:
        data = np.flipud(data)

    # Apply scale and offset if requested.
    if scale != 1.0 or offset != 0.0:
        data = data * scale + offset

    return data
