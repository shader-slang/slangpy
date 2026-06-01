# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading

import pytest

from slangpy import (
    Profiler,
    ProfilerDesc,
    ProfilerTimeline,
    ProfilerTimelineType,
    ProfilerZoneFlags,
    current_profiler,
    current_profiler_or_null,
    pop_current_profiler,
    push_current_profiler,
)


def test_profiler_context_manages_tls_stack():
    profiler = Profiler()

    assert current_profiler_or_null() is None
    with profiler:
        assert current_profiler() is profiler
    assert current_profiler_or_null() is None

    with pytest.raises(RuntimeError, match="No current profiler"):
        current_profiler()


def test_current_profiler_free_functions_are_lifo():
    profiler_a = Profiler()
    profiler_b = Profiler()

    push_current_profiler(profiler_a)
    assert current_profiler() is profiler_a

    push_current_profiler(profiler_b)
    assert current_profiler() is profiler_b
    assert pop_current_profiler() is profiler_b
    assert current_profiler() is profiler_a
    assert pop_current_profiler() is profiler_a
    assert current_profiler_or_null() is None

    with pytest.raises(RuntimeError, match="No current profiler"):
        current_profiler()


def test_current_profiler_is_thread_local():
    main_profiler = Profiler()
    thread_profiler = Profiler()
    thread_ready = threading.Event()

    def thread_main() -> None:
        with thread_profiler:
            assert current_profiler() is thread_profiler
            thread_ready.set()

    with main_profiler:
        thread = threading.Thread(target=thread_main)
        thread.start()
        thread.join()
        assert current_profiler() is main_profiler

    assert thread_ready.is_set()
    assert current_profiler_or_null() is None


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

    assert isinstance(profiler.desc, ProfilerDesc)


def test_profiler_timeline_public_shape():
    timeline = ProfilerTimeline()
    timeline.timeline_id = 5
    timeline.type = ProfilerTimelineType.gpu
    timeline.name = "GPU"

    assert timeline.timeline_id == 5
    assert timeline.type == ProfilerTimelineType.gpu
    assert timeline.name == "GPU"
    assert timeline.queue is not None
    assert ProfilerZoneFlags.gpu & ProfilerZoneFlags.gpu


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
