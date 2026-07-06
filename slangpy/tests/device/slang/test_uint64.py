# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers

ELEMENT_COUNT = 1024


@pytest.mark.parametrize("view", ["uav", "srv"])
@pytest.mark.parametrize(
    "profile",
    [
        "sm_6_0",
        "sm_6_1",
        "sm_6_2",
        "sm_6_3",
        "sm_6_4",
        "sm_6_5",
        "sm_6_6",
        "sm_6_7",
    ],
)
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_uint64(device_type: spy.DeviceType, profile: str | None, view: str):
    if device_type in (spy.DeviceType.cuda, spy.DeviceType.metal):
        if profile != "sm_6_0":
            pytest.skip("Target profiles do not apply to this backend")
        profile = None

    device = helpers.get_device(device_type)

    np.random.seed(123)
    data = np.random.rand(ELEMENT_COUNT).astype(np.uint64)

    ctx = helpers.dispatch_compute(
        device=device,
        path="test_uint64.slang",
        entry_point=f"main_{view}",
        profile=profile,
        thread_count=[ELEMENT_COUNT, 1, 1],
        buffers={
            "data": {"data": data},
            "result": {"element_count": ELEMENT_COUNT * 2},
        },
    )

    result = ctx.buffers["result"].to_numpy().view(np.uint64).flatten()
    assert np.all(result == data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
