# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers

ELEMENT_COUNT = 1024


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_conditional(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    main_module = device.load_module("test_conditional.slang")

    for enable_float in [False, True]:
        link_module = device.load_module_from_source(
            f"config_{str(enable_float).lower()}",
            f"export static const bool ENABLE_FLOAT = {str(enable_float).lower()};",
        )
        program = device.link_program(
            [main_module, link_module], [main_module.entry_point("compute_main")]
        )
        kernel = device.create_compute_kernel(program)

        result_buffer = device.create_buffer(
            size=ELEMENT_COUNT * 4, usage=spy.BufferUsage.unordered_access
        )

        kernel.dispatch(thread_count=[ELEMENT_COUNT, 1, 1], result=result_buffer)

        dtype = np.float32 if enable_float else np.int32

        expected = np.linspace(0, ELEMENT_COUNT - 1, ELEMENT_COUNT, dtype=dtype)
        result = result_buffer.to_numpy().view(dtype).flatten()
        assert np.all(result == expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
