# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from pathlib import Path

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.skip("Skipped because it needs Aftermath which is not enabled by default")
def test_aftermath_tdr(device_type: spy.DeviceType):
    device = spy.Device(
        type=device_type,
        enable_aftermath=True,
        enable_debug_layers=False,
        compiler_options={
            "include_paths": [Path(__file__).parent],
            "debug_info": spy.SlangDebugInfoLevel.maximal,
        },
    )

    program = device.load_program("test_aftermath_tdr.slang", ["compute_main"])
    pipeline = device.create_compute_pipeline(program)
    buffer = device.create_buffer(size=1024, usage=spy.BufferUsage.unordered_access, label="buffer")

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    root_object = compute_pass.bind_pipeline(pipeline)
    cursor = spy.ShaderCursor(root_object).find_entry_point(0)
    cursor["buffer"] = buffer
    compute_pass.dispatch_compute(thread_group_count=(1, 1, 1))
    compute_pass.end()
    device.submit_command_buffer(encoder.finish())
    device.wait()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
