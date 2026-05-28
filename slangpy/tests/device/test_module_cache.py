# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from pathlib import Path

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_cache(device_type: spy.DeviceType, tmpdir: str):
    cache_dir = tmpdir
    # Create device with a module cache.
    device = spy.Device(
        type=device_type,
        enable_print=True,
        module_cache_path=cache_dir,
        compiler_options={"include_paths": [Path(__file__).parent]},
        label=f"module-cache-1-{device_type.name}",
    )
    # Create and dispatch kernel, module should be stored to the cache.
    program = device.load_program(
        module_name="test_module_cache", entry_point_names=["compute_main"]
    )
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(thread_count=[1, 1, 1])
    assert device.flush_print_to_string().strip() == "Hello module cache!"
    # Close device.
    device.close()

    # Re-create device using same module cache location.
    device = spy.Device(
        type=device_type,
        enable_print=True,
        module_cache_path=cache_dir,
        compiler_options={"include_paths": [Path(__file__).parent]},
        label=f"module-cache-1-{device_type.name}",
    )
    # Create and dispatch kernel, shader should be loaded from cache.
    program = device.load_program(
        module_name="test_module_cache", entry_point_names=["compute_main"]
    )
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(thread_count=[1, 1, 1])
    assert device.flush_print_to_string().strip() == "Hello module cache!"
    # Close device.
    device.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
