# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading
from pathlib import Path

import pytest

from slangpy import (
    DeviceType,
    Feature,
    Profiler,
    ProfilerDesc,
    ProfilerTrace,
    ProfilerTimelineInfo,
    ProfilerTimelineType,
    ProfilerZoneFlags,
    current_profiler,
    current_profiler_or_null,
    pop_current_profiler,
    push_current_profiler,
)
from slangpy.testing import helpers


def test_profiler_context_manages_application_stack():
    assert current_profiler_or_null() is None

    profiler = Profiler()
    assert current_profiler() is profiler

    with profiler:
        assert current_profiler() is profiler

    assert current_profiler() is profiler
    del profiler
    assert current_profiler_or_null() is None

    with pytest.raises(RuntimeError, match="No current profiler"):
        current_profiler()


def test_current_profiler_free_functions_are_lifo():
    profiler_a = Profiler()
    assert current_profiler() is profiler_a

    profiler_b = Profiler()
    assert current_profiler() is profiler_b

    push_current_profiler(profiler_a)
    assert current_profiler() is profiler_a

    push_current_profiler(profiler_b)
    assert current_profiler() is profiler_b
    assert pop_current_profiler() is profiler_b
    assert current_profiler() is profiler_a
    assert pop_current_profiler() is profiler_a
    assert current_profiler() is profiler_b

    del profiler_b
    assert current_profiler() is profiler_a

    del profiler_a
    assert current_profiler_or_null() is None

    with pytest.raises(RuntimeError, match="No current profiler"):
        current_profiler()


def test_profiler_destruction_removes_stack_entries():
    profiler = Profiler()
    push_current_profiler(profiler)
    assert current_profiler() is profiler

    del profiler
    assert current_profiler_or_null() is None


def test_current_profiler_is_application_wide():
    profiler = Profiler()
    thread_ready = threading.Event()
    worker_saw_profiler = False

    def thread_main() -> None:
        nonlocal worker_saw_profiler
        worker_saw_profiler = current_profiler() is profiler
        thread_ready.set()

    thread = threading.Thread(target=thread_main)
    thread.start()
    thread.join()
    assert current_profiler() is profiler

    assert thread_ready.is_set()
    assert worker_saw_profiler
    del profiler
    assert current_profiler_or_null() is None


def test_profiler_descriptor_defaults_and_validation():
    desc = ProfilerDesc()
    assert desc.frame_stats_enabled
    assert not desc.trace_enabled_on_start
    assert desc.stats_window_size == 120
    assert desc.gpu_query_pool_size == 64 * 1024
    assert desc.gpu_query_block_size == 256
    assert desc.auto_zones_enabled
    assert not desc.debug_groups_enabled

    desc.stats_window_size = 0
    with pytest.raises(RuntimeError):
        Profiler(desc)

    desc = ProfilerDesc()
    desc.gpu_query_pool_size = 0
    with pytest.raises(RuntimeError):
        Profiler(desc)

    desc = ProfilerDesc()
    desc.gpu_query_block_size = 3
    with pytest.raises(RuntimeError):
        Profiler(desc)

    desc = ProfilerDesc()
    desc.gpu_query_block_size = desc.gpu_query_pool_size + 2
    with pytest.raises(RuntimeError):
        Profiler(desc)


def test_profiler_retained_settings_are_mutable():
    profiler = Profiler()

    assert profiler.enabled
    profiler.enabled = False
    assert not profiler.enabled

    assert profiler.auto_zones_enabled
    profiler.auto_zones_enabled = False
    assert not profiler.auto_zones_enabled

    assert not profiler.debug_groups_enabled
    profiler.debug_groups_enabled = True
    assert profiler.debug_groups_enabled

    assert profiler.frame_stats_enabled
    profiler.frame_stats_enabled = False
    assert not profiler.frame_stats_enabled

    assert profiler.stats_window_size == 120
    profiler.stats_window_size = 4
    assert profiler.stats_window_size == 4
    with pytest.raises(RuntimeError):
        profiler.stats_window_size = 0

    assert isinstance(profiler.desc, ProfilerDesc)
    del profiler
    assert current_profiler_or_null() is None


def test_profiler_timeline_public_shape():
    timeline = ProfilerTimelineInfo()
    timeline.type = ProfilerTimelineType.gpu
    timeline.name = "GPU"

    assert timeline.type == ProfilerTimelineType.gpu
    assert timeline.name == "GPU"
    assert timeline.queue is not None
    assert ProfilerZoneFlags.copy_name & ProfilerZoneFlags.copy_name


def test_profiler_manual_python_zone_and_frame_contexts_are_not_bound():
    profiler = Profiler()

    assert not hasattr(profiler, "zone")
    assert not hasattr(profiler, "frame")

    del profiler
    assert current_profiler_or_null() is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slangpy_auto_gpu_zone_trace(device_type: DeviceType, tmp_path: Path) -> None:
    device = helpers.get_device(device_type)
    if not device.has_feature(Feature.timestamp_query) or not device.has_feature(
        Feature.timestamp_calibration
    ):
        pytest.skip("Timestamp queries with calibration are not supported")

    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
float add_numbers(float a, float b) {
    return a + b;
}
""",
    )

    profiler = Profiler()
    profiler.start_trace()
    try:
        function(1.0, 2.0)
        device.wait()
        profiler.tick()

        trace = profiler.trace_snapshot()
        assert isinstance(trace, ProfilerTrace)

        path = tmp_path / "slangpy-auto-zone.json"
        trace.write_to_json(path)
        text = path.read_text()

        assert "add_numbers" in text
        assert '"cat":"sgl.cpu"' in text
        assert '"cat":"sgl.gpu"' in text
    finally:
        del profiler
        assert current_profiler_or_null() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
