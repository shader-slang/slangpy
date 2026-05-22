# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from slangpy import DeviceType
from slangpy.experimental.gridarg import grid
from slangpy.slangpy import Shape
from slangpy.testing import helpers
from slangpy.types import Tensor
from slangpy.types.threadidarg import thread_id


MAX_DISPATCH_THREAD_GROUPS_X = 65535
DEFAULT_GENERATED_THREAD_GROUP_SIZE = 32
EXTRA_THREADS = 17

MODULE = r"""
import "slangpy";

void record_call_id(int index, RWStructuredBuffer<uint> markers, uint row_stride, uint count)
{
    // Mark sentinel logical IDs around the old single-row dispatch limit. If
    // physical dispatch Y is not flattened correctly, the row_stride entries
    // never get hit.
    uint tid = uint(index);
    if (tid == 0)
        markers[0] = 1;
    if (tid == row_stride - 1)
        markers[1] = tid;
    if (tid == row_stride)
        markers[2] = tid;
    if (tid == count - 1)
        markers[3] = tid;
}

void record_thread_id(uint tid, RWStructuredBuffer<uint> markers, uint row_stride, uint count)
{
    // Same sentinels as record_call_id, but fed by slangpy.thread_id() and
    // _thread_count instead of a vectorized grid argument.
    if (tid == 0)
        markers[0] = 1;
    if (tid == row_stride - 1)
        markers[1] = tid;
    if (tid == row_stride)
        markers[2] = tid;
    if (tid == count - 1)
        markers[3] = tid;
}
"""


def _large_dispatch_shape(device, thread_group_size=DEFAULT_GENERATED_THREAD_GROUP_SIZE):
    limits = device.info.limits.max_compute_dispatch_thread_groups
    dispatch_groups_x = min(limits.x, MAX_DISPATCH_THREAD_GROUPS_X)
    if dispatch_groups_x < 1:
        pytest.skip("Device reports no X compute dispatch groups")
    if limits.y < 2:
        pytest.skip("Device cannot dispatch two rows of compute thread groups")

    # One physical dispatch row is the maximum usable X-group count times the
    # generated kernel's X thread-group size. Requesting a few extra threads
    # forces SlangPy to launch a second physical Y row.
    row_stride = dispatch_groups_x * thread_group_size
    count = row_stride + EXTRA_THREADS
    return row_stride, count


def _markers(device, module):
    return Tensor.zeros(device, (4,), dtype=module.layout.find_type_by_name("uint"))


def _read_markers(markers):
    return markers.to_numpy().view(np.uint32).reshape(-1)


def _expected(row_stride, count):
    # markers[0] stores 1 instead of 0 so that "thread 0 ran" is visible in
    # an otherwise zero-initialized buffer.
    return np.array([1, row_stride - 1, row_stride, count - 1], dtype=np.uint32)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_large_vectorized_grid_dispatch(device_type: DeviceType):
    device = helpers.get_device(device_type)
    row_stride, count = _large_dispatch_shape(device)
    module = helpers.create_module(device, MODULE)
    markers = _markers(device, module)

    # Exercise the normal vectorized path, where the call shape comes from
    # grid((count,)).
    module.record_call_id(grid((count,)), markers.storage, row_stride, count)

    np.testing.assert_array_equal(_read_markers(markers), _expected(row_stride, count))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_large_vectorized_call_group_dispatch(device_type: DeviceType):
    device = helpers.get_device(device_type)
    row_stride, count = _large_dispatch_shape(device)
    module = helpers.create_module(device, MODULE)
    markers = _markers(device, module)

    # Exercise the call-group path, which also has to flatten SV_GroupID.y
    # into the logical flat group ID used by callshape.slang.
    module.record_call_id.call_group_shape(Shape((DEFAULT_GENERATED_THREAD_GROUP_SIZE,)))(
        grid((count,)), markers.storage, row_stride, count
    )

    np.testing.assert_array_equal(_read_markers(markers), _expected(row_stride, count))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_large_thread_count_dispatch(device_type: DeviceType):
    device = helpers.get_device(device_type)
    row_stride, count = _large_dispatch_shape(device)
    module = helpers.create_module(device, MODULE)
    markers = _markers(device, module)

    # Exercise the explicit _thread_count path, where the shader's logical
    # thread id comes from slangpy.thread_id() rather than a vectorized arg.
    module.record_thread_id(thread_id(), markers.storage, row_stride, count, _thread_count=count)

    np.testing.assert_array_equal(_read_markers(markers), _expected(row_stride, count))
