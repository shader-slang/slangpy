# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from pathlib import Path

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_shader_cache(device_type: spy.DeviceType, tmpdir: str):
    cache_dir = tmpdir
    # Create device with a shader cache.
    device = spy.Device(
        type=device_type,
        enable_print=True,
        shader_cache_path=cache_dir,
        compiler_options={"include_paths": [Path(__file__).parent]},
        label=f"shader-cache-1-{device_type.name}",
    )
    # We expect the cache to be empty and untouched.
    stats = device.shader_cache_stats
    assert stats.entry_count == 0
    assert stats.hit_count == 0
    assert stats.miss_count == 0
    # Create and dispatch kernel, shader should be stored to the cache.
    program = device.load_program(
        module_name="test_shader_cache", entry_point_names=["compute_main"]
    )
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(thread_count=[1, 1, 1])
    assert device.flush_print_to_string().strip() == "Hello shader cache!"
    # We expect at least one entry but potentially more than one
    # (pipelines can get cached in addition to the compiled shader binary).
    # We also expect at least one miss because the cache was empty.
    stats = device.shader_cache_stats
    assert stats.entry_count > 0
    assert stats.hit_count == 0
    assert stats.miss_count > 0
    # Close device.
    device.close()

    # Re-create device using same shader cache location.
    device = spy.Device(
        type=device_type,
        enable_print=True,
        shader_cache_path=cache_dir,
        compiler_options={"include_paths": [Path(__file__).parent]},
        label=f"shader-cache-1-{device_type.name}",
    )
    # We expect at least one entry, but hit/miss count are reset.
    stats = device.shader_cache_stats
    assert stats.entry_count > 0
    assert stats.hit_count == 0
    assert stats.miss_count == 0
    entry_count_before = stats.entry_count
    # Create and dispatch kernel, shader should be loaded from cache.
    program = device.load_program(
        module_name="test_shader_cache", entry_point_names=["compute_main"]
    )
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(thread_count=[1, 1, 1])
    assert device.flush_print_to_string().strip() == "Hello shader cache!"
    # We expect the same number of entries in the cache, but at least one hit.
    stats = device.shader_cache_stats
    assert stats.entry_count == entry_count_before
    assert stats.hit_count > 0
    assert stats.miss_count == 0
    # Close device.
    device.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_shader_cache_additional_source(device_type: spy.DeviceType, tmp_path: Path) -> None:
    for cached in [False, True]:
        device = spy.Device(
            type=device_type,
            enable_print=True,
            shader_cache_path=tmp_path,
            compiler_options={"include_paths": [Path(__file__).parent]},
        )
        try:
            if cached:
                # A different load order must still find the same cached program.
                device.load_program(
                    "test_shader_cache",
                    ["compute_additional"],
                    additional_source="export static const int VALUE = -1;",
                )
            program = device.load_program(
                "test_shader_cache",
                ["compute_additional"],
                additional_source="export static const int VALUE = 7;",
            )
            kernel = device.create_compute_kernel(program)
            kernel.dispatch(thread_count=[1, 1, 1])
            assert device.flush_print_to_string().strip() == "7"
            stats = device.shader_cache_stats
            assert stats.entry_count > 0
            if cached:
                assert stats.hit_count > 0
                assert stats.miss_count == 0
                # Changed source must compile again and produce the new result.
                program = device.load_program(
                    "test_shader_cache",
                    ["compute_additional"],
                    additional_source="export static const int VALUE = 8;",
                )
                kernel = device.create_compute_kernel(program)
                kernel.dispatch(thread_count=[1, 1, 1])
                assert device.flush_print_to_string().strip() == "8"
                assert device.shader_cache_stats.miss_count > 0
            else:
                assert stats.hit_count == 0
                assert stats.miss_count > 0
        finally:
            device.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
