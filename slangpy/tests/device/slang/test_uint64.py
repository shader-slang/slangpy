# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers

ELEMENT_COUNT = 1024

CASES = [
    (spy.DeviceType.d3d12, None),
    (spy.DeviceType.d3d12, "sm_6_0"),
    (spy.DeviceType.d3d12, "sm_6_9"),
    (spy.DeviceType.vulkan, None),
    (spy.DeviceType.vulkan, "spirv_1_3"),
    (spy.DeviceType.vulkan, "spirv_1_6"),
    (spy.DeviceType.cuda, None),
    (spy.DeviceType.metal, None),
    (spy.DeviceType.cpu, None),
    (spy.DeviceType.wgpu, None),
    (spy.DeviceType.automatic, None),
]


@pytest.mark.parametrize("view", ["uav", "srv"])
@pytest.mark.parametrize(
    "device_type, profile",
    [case for case in CASES if case[0] in helpers.DEFAULT_DEVICE_TYPES],
)
def test_uint64(device_type: spy.DeviceType, profile: str | None, view: str) -> None:
    device = helpers.get_device(device_type)
    if profile is not None and "_" + profile not in device.capabilities:
        pytest.skip(f"Device does not advertise {profile}")

    np.random.seed(123)
    data = np.random.rand(ELEMENT_COUNT).astype(np.uint64)

    ctx = helpers.dispatch_compute(
        device=device,
        path="test_uint64.slang",
        entry_point=f"main_{view}",
        compiler_options={"profile": profile} if profile else {},
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
