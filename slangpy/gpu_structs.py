# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""NumPy layouts and field encodings for portable slang-rhi GPU records."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


# Values used to encode fields in triangle_cluster_args_dtype. These mirror
# slang-rhi-device.h and are intentionally scoped to GPU record construction.
CLUSTER_FLAG_NONE = 0
CLUSTER_FLAG_ALLOW_DISABLE_OMMS = 1 << 0

CLUSTER_INDEX_FORMAT_UINT8 = 1
CLUSTER_INDEX_FORMAT_UINT16 = 2
CLUSTER_INDEX_FORMAT_UINT32 = 4

CLUSTER_GEOMETRY_FLAG_NONE = 0
CLUSTER_GEOMETRY_FLAG_CULL_DISABLE = 1 << 29
CLUSTER_GEOMETRY_FLAG_NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 30
CLUSTER_GEOMETRY_FLAG_OPAQUE = 1 << 31


def _structured_dtype(
    names: list[str],
    formats: list[str],
    offsets: list[int],
    itemsize: int,
) -> np.dtype[Any]:
    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": itemsize,
        }
    )


indirect_draw_arguments_dtype = _structured_dtype(
    [
        "vertex_count_per_instance",
        "instance_count",
        "start_vertex_location",
        "start_instance_location",
    ],
    ["<u4", "<u4", "<u4", "<u4"],
    [0, 4, 8, 12],
    16,
)

indirect_draw_indexed_arguments_dtype = _structured_dtype(
    [
        "index_count_per_instance",
        "instance_count",
        "start_index_location",
        "base_vertex_location",
        "start_instance_location",
    ],
    ["<u4", "<u4", "<u4", "<i4", "<u4"],
    [0, 4, 8, 12, 16],
    20,
)

indirect_dispatch_arguments_dtype = _structured_dtype(
    ["thread_group_count_x", "thread_group_count_y", "thread_group_count_z"],
    ["<u4", "<u4", "<u4"],
    [0, 4, 8],
    12,
)

aabb_dtype = _structured_dtype(
    ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z"],
    ["<f4", "<f4", "<f4", "<f4", "<f4", "<f4"],
    [0, 4, 8, 12, 16, 20],
    24,
)

micromap_triangle_desc_dtype = _structured_dtype(
    ["data_offset", "subdivision_level", "format"],
    ["<u4", "<u2", "<u2"],
    [0, 4, 6],
    8,
)

# NumPy has no bitfield dtype. The five bitfields in rhi::TriangleClusterArgs
# therefore share one explicitly packed uint32 field at byte offset 8.
triangle_cluster_args_dtype = _structured_dtype(
    [
        "cluster_id",
        "cluster_flags",
        "packed_counts_and_formats",
        "base_geometry_index_and_flags",
        "index_buffer_stride",
        "vertex_buffer_stride",
        "geometry_index_and_flags_buffer_stride",
        "opacity_micromap_index_buffer_stride",
        "index_buffer",
        "vertex_buffer",
        "geometry_index_and_flags_buffer",
        "opacity_micromap_array",
        "opacity_micromap_index_buffer",
        "instantiation_bounding_box_limit",
    ],
    [
        "<u4",
        "<u4",
        "<u4",
        "<u4",
        "<u2",
        "<u2",
        "<u2",
        "<u2",
        "<u8",
        "<u8",
        "<u8",
        "<u8",
        "<u8",
        "<u8",
    ],
    [0, 4, 8, 12, 16, 18, 20, 22, 24, 32, 40, 48, 56, 64],
    72,
)

instantiate_template_args_dtype = _structured_dtype(
    [
        "cluster_id_offset",
        "geometry_index_offset",
        "cluster_template",
        "vertex_buffer",
        "vertex_buffer_stride",
    ],
    ["<u4", "<u4", "<u8", "<u8", "<u8"],
    [0, 4, 8, 16, 24],
    32,
)

cluster_args_dtype = _structured_dtype(
    ["cluster_handles_count", "cluster_handles_stride", "cluster_handles_buffer"],
    ["<u4", "<u4", "<u8"],
    [0, 4, 8],
    16,
)


def _checked_uint_field(value: npt.ArrayLike, name: str, maximum: int) -> npt.NDArray[np.uint32]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if np.any(array < 0) or np.any(array > maximum):
        raise ValueError(f"{name} must be in the range [0, {maximum}]")
    return array.astype(np.uint32, copy=False)


def pack_triangle_cluster_args_fields(
    triangle_count: npt.ArrayLike,
    vertex_count: npt.ArrayLike,
    position_truncate_bit_count: npt.ArrayLike = 0,
    index_format: npt.ArrayLike = 0,
    opacity_micromap_index_format: npt.ArrayLike = 0,
) -> npt.NDArray[np.uint32]:
    """
    Pack the bitfields stored in ``triangle_cluster_args_dtype``.

    Arguments may be integer scalars or broadcast-compatible integer arrays.
    Cluster triangle and vertex counts are limited to the portable slang-rhi
    maximum of 256.
    """

    triangles = _checked_uint_field(triangle_count, "triangle_count", 256)
    vertices = _checked_uint_field(vertex_count, "vertex_count", 256)
    truncate = _checked_uint_field(position_truncate_bit_count, "position_truncate_bit_count", 0x3F)
    indices = _checked_uint_field(index_format, "index_format", 0xF)
    omm_indices = _checked_uint_field(
        opacity_micromap_index_format,
        "opacity_micromap_index_format",
        0xF,
    )
    return (
        triangles
        | (vertices << np.uint32(9))
        | (truncate << np.uint32(18))
        | (indices << np.uint32(24))
        | (omm_indices << np.uint32(28))
    )


__all__ = [
    "aabb_dtype",
    "cluster_args_dtype",
    "CLUSTER_FLAG_ALLOW_DISABLE_OMMS",
    "CLUSTER_FLAG_NONE",
    "CLUSTER_GEOMETRY_FLAG_CULL_DISABLE",
    "CLUSTER_GEOMETRY_FLAG_NO_DUPLICATE_ANY_HIT_INVOCATION",
    "CLUSTER_GEOMETRY_FLAG_NONE",
    "CLUSTER_GEOMETRY_FLAG_OPAQUE",
    "CLUSTER_INDEX_FORMAT_UINT16",
    "CLUSTER_INDEX_FORMAT_UINT32",
    "CLUSTER_INDEX_FORMAT_UINT8",
    "indirect_dispatch_arguments_dtype",
    "indirect_draw_arguments_dtype",
    "indirect_draw_indexed_arguments_dtype",
    "instantiate_template_args_dtype",
    "micromap_triangle_desc_dtype",
    "pack_triangle_cluster_args_fields",
    "triangle_cluster_args_dtype",
]
