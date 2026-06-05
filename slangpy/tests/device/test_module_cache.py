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

    # Check that cached binary module files were created.
    cache_root = Path(cache_dir)
    slang_module_files = list(cache_root.rglob("*.slang-module"))
    assert len(slang_module_files) > 0, "Expected cached .slang-module files"

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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_source_module_cache(device_type: spy.DeviceType, tmpdir: str):
    cache_dir = tmpdir
    source = """
RWStructuredBuffer<float> output;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main() {
    output[0] = 42.0;
}
"""
    # Create device with a module cache.
    device = spy.Device(
        type=device_type,
        module_cache_path=cache_dir,
        label=f"source-module-cache-1-{device_type.name}",
    )
    # Load module from source, link program, and dispatch.
    module = device.load_module_from_source("test_source_cache", source)
    ep = module.entry_point("compute_main")
    program = device.link_program([module], [ep])
    kernel = device.create_compute_kernel(program)
    output = device.create_buffer(
        element_count=1,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    kernel.dispatch(thread_count=[1, 1, 1], vars={"output": output})
    import numpy as np

    result = np.frombuffer(output.to_numpy(), dtype=np.float32)
    assert result[0] == 42.0

    # Check that cache files were created under source_modules directory.
    cache_root = Path(cache_dir)
    slang_module_files = list(cache_root.rglob("source_modules/*.slang-module"))
    assert len(slang_module_files) > 0, "Expected cached .slang-module files for source module"

    # Close device.
    device.close()

    # Re-create device using same module cache location.
    device = spy.Device(
        type=device_type,
        module_cache_path=cache_dir,
        label=f"source-module-cache-2-{device_type.name}",
    )
    # Load same source again - should load from cache.
    module = device.load_module_from_source("test_source_cache", source)
    ep = module.entry_point("compute_main")
    program = device.link_program([module], [ep])
    kernel = device.create_compute_kernel(program)
    output = device.create_buffer(
        element_count=1,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    kernel.dispatch(thread_count=[1, 1, 1], vars={"output": output})
    result = np.frombuffer(output.to_numpy(), dtype=np.float32)
    assert result[0] == 42.0
    # Close device.
    device.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
