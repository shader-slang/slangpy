# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import pytest
import slangpy as spy
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import sglhelpers as helpers

ELEMENT_COUNT = 1024


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_cast_float16(device_type: spy.DeviceType):
    if device_type == spy.DeviceType.metal:
        pytest.skip("float16 cast not supported on Metal")

    device = helpers.get_device(device_type)

    np.random.seed(123)
    data = np.random.rand(ELEMENT_COUNT, 2).astype(np.float16)

    ctx = helpers.dispatch_compute(
        device=device,
        path=Path(__file__).parent / "test_cast_float16.slang",
        entry_point="compute_main",
        thread_count=[ELEMENT_COUNT, 1, 1],
        buffers={
            "data": {"data": data},
            "result": {"element_count": ELEMENT_COUNT},
        },
    )

    expected = data.view(np.uint32).flatten()
    result = ctx.buffers["result"].to_numpy().view(np.uint32).flatten()
    assert np.all(result == expected)


if __name__ == "__main__":
    pytest.main([__file__, "-vvvs"])
