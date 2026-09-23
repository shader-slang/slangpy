# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers

ELEMENT_COUNT = 1024


@pytest.mark.parametrize("view", ["uav", "srv"])
@pytest.mark.parametrize(
    "device_type, profile",
    [
        (device_type, profile)
        for device_type in helpers.DEFAULT_DEVICE_TYPES
        for profile in (
            [None] + [f"sm_6_{minor}" for minor in range(2, 10)]
            if device_type == spy.DeviceType.d3d12
            else (
                [None, "spirv_1_3", "spirv_1_5", "spirv_1_6"]
                if device_type == spy.DeviceType.vulkan
                else [None]
            )
        )
    ],
)
def test_float16(device_type: spy.DeviceType, profile: str | None, view: str) -> None:
    device = helpers.get_device(device_type)
    if profile is not None and "_" + profile not in device.capabilities:
        pytest.skip(f"Device does not advertise {profile}")

    np.random.seed(123)
    data = np.random.rand(ELEMENT_COUNT).astype(np.float16)

    ctx = helpers.dispatch_compute(
        device=device,
        path="test_float16.slang",
        entry_point=f"main_{view}",
        compiler_options={"profile": profile, "capabilities": []} if profile else {},
        thread_count=[ELEMENT_COUNT, 1, 1],
        buffers={
            "data": {"data": data},
            "result": {"element_count": ELEMENT_COUNT},
        },
    )

    result = ctx.buffers["result"].to_numpy().view(np.float16).flatten()
    assert np.all(result == data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
