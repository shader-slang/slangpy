# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import pytest
import slangpy as spy
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import sglhelpers as helpers

ELEMENT_COUNT = 1024


@pytest.mark.parametrize("view", ["uav", "srv"])
@pytest.mark.parametrize("shader_model", helpers.all_shader_models_from(spy.ShaderModel.sm_6_2))
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_float16(device_type: spy.DeviceType, shader_model: spy.ShaderModel, view: str):
    if device_type == spy.DeviceType.cuda and (sys.platform == "linux" or sys.platform == "linux2"):
        pytest.skip(
            "Slang fails to find cuda_fp16.h header https://github.com/shader-slang/slang/issues/7037"
        )

    device = helpers.get_device(device_type)

    np.random.seed(123)
    data = np.random.rand(ELEMENT_COUNT).astype(np.float16)

    ctx = helpers.dispatch_compute(
        device=device,
        path=Path(__file__).parent / "test_float16.slang",
        entry_point=f"main_{view}",
        shader_model=shader_model,
        thread_count=[ELEMENT_COUNT, 1, 1],
        buffers={
            "data": {"data": data},
            "result": {"element_count": ELEMENT_COUNT},
        },
    )

    result = ctx.buffers["result"].to_numpy().view(np.float16).flatten()
    assert np.all(result == data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
