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
    ProfilerTimelineType,
    ProfilerZoneFlags,
    current_profiler,
    current_profiler_or_null,
    pop_current_profiler,
    push_current_profiler,
)
from slangpy.testing import helpers


def test_profiler_context_manages_thread_stack():
    assert current_profiler_or_null() is None

    profiler = Profiler()
    assert current_profiler_or_null() is None

    with profiler:
        assert current_profiler() is profiler

    assert current_profiler_or_null() is None
    del profiler
    assert current_profiler_or_null() is None

    with pytest.raises(RuntimeError, match="No current profiler"):
        current_profiler()


def test_current_profiler_free_functions_are_lifo():
    profiler_a = Profiler()
    assert current_profiler_or_null() is None

    profiler_b = Profiler()
    assert current_profiler_or_null() is None

    push_current_profiler(profiler_a)
    assert current_profiler() is profiler_a

    push_current_profiler(profiler_b)
    assert current_profiler() is profiler_b
    assert pop_current_profiler() is profiler_b
    assert current_profiler() is profiler_a

    push_current_profiler(profiler_a)
    assert current_profiler() is profiler_a
    assert pop_current_profiler() is profiler_a
    assert current_profiler() is profiler_a
    assert pop_current_profiler() is profiler_a

    del profiler_b
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


def test_current_profiler_is_thread_local():
    profiler = Profiler()
    thread_ready = threading.Event()
    worker_saw_profiler = False

    def thread_main() -> None:
        nonlocal worker_saw_profiler
        worker_saw_profiler = current_profiler_or_null() is profiler
        thread_ready.set()

    with profiler:
        thread = threading.Thread(target=thread_main)
        thread.start()
        thread.join()
        assert current_profiler() is profiler

    assert thread_ready.is_set()
    assert not worker_saw_profiler
    del profiler
    assert current_profiler_or_null() is None


def test_profiler_descriptor_defaults_and_validation():
    desc = ProfilerDesc()
    assert not desc.trace_enabled_on_start
    assert desc.stats_window_size == 120
    assert desc.gpu_query_pool_size == 64 * 1024
    assert desc.auto_zones_enabled
    assert not desc.debug_groups_enabled

    desc.stats_window_size = 0
    with pytest.raises(RuntimeError):
        Profiler(desc)

    desc = ProfilerDesc()
    desc.gpu_query_pool_size = 0
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

    assert profiler.stats_window_size == 120
    profiler.stats_window_size = 4
    assert profiler.stats_window_size == 4
    with pytest.raises(RuntimeError):
        profiler.stats_window_size = 0

    assert isinstance(profiler.desc, ProfilerDesc)
    del profiler
    assert current_profiler_or_null() is None


def test_profiler_public_flags_are_bound():
    assert ProfilerTimelineType.gpu is not None
    assert ProfilerZoneFlags.copy_name & ProfilerZoneFlags.copy_name


def test_profiler_manual_python_zone_and_frame_contexts_are_not_bound():
    profiler = Profiler()

    assert not hasattr(profiler, "zone")
    assert hasattr(profiler, "frame")

    del profiler
    assert current_profiler_or_null() is None


def test_profiler_frame_context_activates_current_thread():
    profiler = Profiler()

    assert current_profiler_or_null() is None
    with profiler.frame("python_frame"):
        assert current_profiler() is profiler
    assert current_profiler_or_null() is None


def test_profiler_nested_frame_context_fails_clearly():
    profiler = Profiler()

    with profiler.frame("outer"):
        with pytest.raises(RuntimeError, match="frame is already active"):
            with profiler.frame("inner"):
                pass


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
        with profiler.frame("python_functional_frame"):
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
