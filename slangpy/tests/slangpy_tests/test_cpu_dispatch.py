# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from slangpy import DeviceType
from slangpy.core.generator import (
    MAX_DISPATCH_THREAD_GROUPS_X,
    resolve_max_dispatch_groups_x,
)
from slangpy.experimental.gridarg import grid
from slangpy.testing import helpers

MODULE = r"""
float sq(int i) { return float(i * i); }
"""


def test_zero_dispatch_group_limit_is_unbounded() -> None:
    # A zero (unset) limit must resolve to the ceiling, not 0 -- a zero stride
    # would collapse the physical->logical group flattening.
    assert resolve_max_dispatch_groups_x(0) == MAX_DISPATCH_THREAD_GROUPS_X
    assert resolve_max_dispatch_groups_x(5) == 5
    assert (
        resolve_max_dispatch_groups_x(MAX_DISPATCH_THREAD_GROUPS_X * 4)
        == MAX_DISPATCH_THREAD_GROUPS_X
    )


@pytest.mark.parametrize("device_type", [DeviceType.cpu])
def test_cpu_dispatch_grid(device_type: DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    # Span many thread groups (> one group of 32) to exercise multi-group X indexing.
    n = 1000
    result = module.sq(grid(shape=(n,)), _result="numpy")

    expected = np.arange(n, dtype=np.float32) ** 2
    np.testing.assert_array_equal(result, expected)
