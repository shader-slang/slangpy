# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


# Tests different ways PrintableString can be returned from a Slang struct.
# Detected error reported here: https://github.com/shader-slang/slang/issues/8694
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_printable_string(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    program = device.load_program("test_printable_string.slang", ["compute_main"])
    kernel = device.create_compute_kernel(program)
    hashed_strings = program.layout.hashed_strings
    hash_to_string = {obj.hash: obj.string for obj in hashed_strings}

    result_buffer = device.create_buffer(
        resource_type_layout=kernel.reflection.result,
        element_count=5,
        usage=spy.BufferUsage.unordered_access,
    )

    kernel.dispatch(thread_count=[1, 1, 1], vars={"result": result_buffer})

    result = result_buffer.to_numpy().view(np.uint32).flatten()

    assert hash_to_string[result[0]] == "string_from_function"
    assert hash_to_string[result[1]] == "string_from_method"
    # Disabled due to: https://github.com/shader-slang/slang/issues/8694
    # assert hash_to_string[result[2]] == "string_from_static_const"
    # assert hash_to_string[result[3]] == "string_from_interface_static_const"
    assert hash_to_string[result[4]] == "string_from_interface_method"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
