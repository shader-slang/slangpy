# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import numpy as np

import slangpy as spy
from slangpy.testing import helpers

ELEMENT_COUNT = 1024


@pytest.mark.parametrize("view", ["uav", "srv"])
@pytest.mark.parametrize(
    "shader_model",
    [
        spy.ShaderModel.sm_6_0,
        spy.ShaderModel.sm_6_1,
        spy.ShaderModel.sm_6_2,
        spy.ShaderModel.sm_6_3,
        spy.ShaderModel.sm_6_4,
        spy.ShaderModel.sm_6_5,
        spy.ShaderModel.sm_6_6,
        spy.ShaderModel.sm_6_7,
    ],
)
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_float64(device_type: spy.DeviceType, shader_model: spy.ShaderModel, view: str):
    if sys.platform == "darwin":
        pytest.skip("Not supported on macOS")

    device = helpers.get_device(device_type)

    np.random.seed(123)
    data = np.random.rand(ELEMENT_COUNT).astype(np.float64)

    ctx = helpers.dispatch_compute(
        device=device,
        path="test_float64.slang",
        entry_point=f"main_{view}",
        shader_model=shader_model,
        thread_count=[ELEMENT_COUNT, 1, 1],
        buffers={
            "data": {"data": data},
            "result": {"element_count": ELEMENT_COUNT},
        },
    )

    result = ctx.buffers["result"].to_numpy().view(np.float64).flatten()
    assert np.all(result == data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
