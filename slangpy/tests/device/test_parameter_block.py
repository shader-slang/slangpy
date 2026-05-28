# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


# The shader code is in test_parameterBlock.slang, fill in the parameter block
# 'inputStruct' for the field 'a = 1.0', b = 2, c = 3, then read back the result
# 'd' from the buffer and assert it equals 6.0f. The test will only launch 1 thread
# to test the ParameterBlock binding.
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_parameter_block(device_type: spy.DeviceType):
    if device_type == spy.DeviceType.metal:
        pytest.skip("Crash in slang-rhi due to invalid reflection data")

    # Create device
    device = helpers.get_device(type=device_type)

    if not device.has_feature(spy.Feature.parameter_block):
        pytest.skip("Parameter block feature not supported on this device.")

    # Load the shader program
    program = device.load_program("test_parameter_block.slang", ["computeMain"])
    kernel = device.create_compute_kernel(program)

    # Create a buffer for the output
    output_buffer = device.create_buffer(
        element_count=1,  # Only need one element as we're only launching one thread
        struct_size=4,  # float is 4 bytes
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
    )

    input_buffer = device.create_buffer(
        element_count=1,  # Only need one element as we're only launching one thread
        struct_size=4,  # float is 4 bytes
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
        data=np.array([6.0], dtype=np.float32),
    )

    # Encode compute commands
    command_encoder = device.create_command_encoder()
    with command_encoder.begin_compute_pass() as encoder:
        # Bind the pipeline
        shader_object = encoder.bind_pipeline(kernel.pipeline)

        # Create a shader cursor for the parameter block
        cursor = spy.ShaderCursor(shader_object)

        # Fill in the parameter block values
        cursor["input_struct"]["a"]["aa"] = 1.0
        cursor["input_struct"]["a"]["bb"] = 2
        cursor["input_struct"]["b"] = 3
        cursor["input_struct"]["c"] = output_buffer

        cursor["input_struct"]["nest_param_block"]["aa"] = 4.0
        cursor["input_struct"]["nest_param_block"]["bb"] = 5
        cursor["input_struct"]["nest_param_block"]["nc"] = input_buffer

        # Dispatch a single thread
        encoder.dispatch(thread_count=[1, 1, 1])

    # Submit the command buffer
    device.submit_command_buffer(command_encoder.finish())

    # Read back the result
    result = output_buffer.to_numpy().view(np.float32)[0]

    # Verify the result
    assert result == 21.0, f"Expected 21.0, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
