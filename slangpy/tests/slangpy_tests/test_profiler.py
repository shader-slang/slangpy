# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading
from pathlib import Path

import numpy as np
import pytest

import slangpy as spy
from slangpy.testing import helpers


def record_nested_zones() -> None:
    with spy.profile_frame("frame"):
        with spy.profile_zone("outer"):
            with spy.profile_zone("inner"):
                pass


def record_ambient_profile_function() -> None:
    with spy.profile_function():
        pass


class AmbientProfileOwner:
    def record(self) -> None:
        with spy.profile_function():
            pass


def record_named_zone(name: str) -> None:
    with spy.profile_zone(name):
        pass


def record_same_line_zones() -> None:
    with spy.profile_zone("same-line"), spy.profile_zone("same-line"):
        pass


def test_profiler_configuration() -> None:
    desc = spy.ProfilerDesc(
        frame_stats_window_size=17,
        enable_auto_zones=False,
        enable_debug_groups=True,
    )
    assert desc.frame_stats_window_size == 17
    assert not desc.enable_auto_zones
    assert desc.enable_debug_groups

    profiler = spy.Profiler(desc)
    assert not profiler.enable_auto_zones
    assert profiler.enable_debug_groups
    profiler.enable_auto_zones = True
    profiler.enable_debug_groups = False
    assert profiler.enable_auto_zones
    assert not profiler.enable_debug_groups


def test_current_profiler_stack_and_thread_locality() -> None:
    def exercise_profiler_stack() -> None:
        profiler = spy.Profiler()
        assert spy.current_profiler() is profiler

        explicit_profiler = spy.Profiler()
        assert spy.current_profiler() is profiler
        with explicit_profiler:
            assert spy.current_profiler() is explicit_profiler
            with spy.profile_frame("selected frame"):
                with spy.profile_zone("selected zone"):
                    assert spy.current_profiler() is explicit_profiler
        assert spy.current_profiler() is profiler

        seen: list[object | None] = []
        worker = threading.Thread(target=lambda: seen.append(spy.current_profiler_or_null()))
        worker.start()
        worker.join()
        assert seen == [None]

        with spy.profile_zone("default zone"):
            assert spy.current_profiler() is profiler
        with spy.profile_frame("default frame"):
            assert spy.current_profiler() is profiler

    exercise_profiler_stack()
    assert spy.current_profiler_or_null() is None


def test_ambient_profile_contexts() -> None:
    profiler = spy.Profiler()
    profiler.start_capture()

    def record_without_current_profiler() -> None:
        with spy.profile_frame("inactive frame"):
            with spy.profile_zone("inactive"):
                record_ambient_profile_function()

    worker = threading.Thread(target=record_without_current_profiler)
    worker.start()
    worker.join()

    with spy.profile_frame("ambient frame"):
        with spy.profile_zone("ambient zone"):
            record_ambient_profile_function()
            AmbientProfileOwner().record()

    trace = profiler.stop_capture()
    assert len(trace.frames) == 1
    assert trace.query_zones(name="inactive").count == 0
    assert trace.query_zones(name="ambient zone").count == 1
    assert trace.query_zones(name="record_ambient_profile_function").count == 1
    assert trace.query_zones(name="AmbientProfileOwner.record").count == 1

    function_site = next(
        site for site in trace.sites if site.name == "record_ambient_profile_function"
    )
    assert Path(function_site.file).name == "test_profiler.py"
    assert function_site.function.endswith("record_ambient_profile_function")


def test_python_sites_use_visible_callsite_metadata() -> None:
    profiler = spy.Profiler()
    profiler.start_capture()

    for _ in range(3):
        record_named_zone("repeated zone")
    record_same_line_zones()

    for _ in range(2):
        with spy.profile_frame("repeated frame"):
            pass

    trace = profiler.stop_capture()
    sites = {site.id: site for site in trace.sites}
    zone_site_ids = [site_id for chunk in trace.zone_chunks for site_id in chunk.site_id.tolist()]
    repeated_site_ids = [
        site_id for site_id in zone_site_ids if sites[site_id].name == "repeated zone"
    ]
    same_line_site_ids = [
        site_id for site_id in zone_site_ids if sites[site_id].name == "same-line"
    ]

    assert len(repeated_site_ids) == 3
    assert len(set(repeated_site_ids)) == 1
    assert len(same_line_site_ids) == 2
    assert len(set(same_line_site_ids)) == 1
    assert len(trace.frames) == 2
    assert trace.frames[0].site_id == trace.frames[1].site_id
    assert trace.timelines[0].thread_id == threading.get_native_id()


def test_capture_hierarchy_queries_and_statistics() -> None:
    profiler = spy.Profiler()
    profiler.start_capture()
    record_nested_zones()
    trace = profiler.stop_capture()

    assert trace.zone_count == 2
    assert len(trace.frames) == 1
    assert trace.stop_reason == spy.ProfilerCaptureStopReason.user
    assert not trace.truncated
    assert trace.diagnostics.producer_drop_count == 0
    assert trace.diagnostics.thread_event_queue_high_water_mark >= 1

    inner = trace.query_zones(name="inner")
    assert inner.count == 1
    statistics = inner.statistics()
    assert statistics.count == 1
    assert statistics.total_ms >= 0.0
    assert statistics.minimum_ms == statistics.maximum_ms
    assert statistics.p50_ms == statistics.mean_ms

    chunks = trace.zone_chunks
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start_ns.dtype == np.uint64
    assert chunk.duration_ns.dtype == np.uint64
    assert chunk.timeline_id.dtype == np.uint32
    assert chunk.parent_index.dtype == np.int32
    assert not chunk.start_ns.flags.writeable
    assert sorted(chunk.parent_index.tolist()) == [-1, 1]


def test_live_and_frame_stats_snapshots_are_immutable() -> None:
    profiler = spy.Profiler()
    with spy.profile_frame():
        with spy.profile_zone("first"):
            pass
    profiler.flush()
    first = profiler.live_snapshot()
    assert first.zone_count == 1
    first_stats = profiler.frame_stats_snapshot()
    assert [entry.name for entry in first_stats.entries] == ["first"]
    assert first_stats.sample_count == 1
    first_sample_index = first_stats.sample_frame_index[0]
    assert first_stats.sample_call_count.tolist() == [[1]]

    with spy.profile_frame():
        with spy.profile_zone("second"):
            pass
    profiler.flush()
    second = profiler.live_snapshot()
    assert first.zone_count == 1
    assert second.zone_count == 2
    second_stats = profiler.frame_stats_snapshot()
    assert [entry.name for entry in first_stats.entries] == ["first"]
    assert first_stats.sample_frame_index[0] == first_sample_index
    assert first_stats.sample_call_count.tolist() == [[1]]
    assert {entry.name for entry in second_stats.entries} == {"first", "second"}
    profiler.clear_frame_stats()
    assert profiler.frame_stats_snapshot().sample_count == 0
    assert first_stats.sample_call_count.tolist() == [[1]]


def test_statistics_are_keyed_by_hierarchical_site_path() -> None:
    profiler = spy.Profiler()
    with spy.profile_frame():
        for parent in ("parent a", "parent b"):
            with spy.profile_zone(parent):
                with spy.profile_zone("shared child"):
                    pass
    profiler.flush()
    stats = profiler.frame_stats_snapshot()
    entries = stats.entries
    shared = [entry for entry in entries if entry.name == "shared child"]
    assert len(shared) == 2
    assert {entries[entry.parent_index].name for entry in shared} == {
        "parent a",
        "parent b",
    }
    text = str(stats)
    assert text.startswith("ProfilerFrameStats(pending_frame_count=0")
    assert "parent a: calls/frame mean=1" in text
    assert "shared child: calls/frame mean=1" in text
    assert "gpu/frame=n/a" in text


def test_frame_statistics_aggregate_repeated_and_absent_zones() -> None:
    profiler = spy.Profiler(spy.ProfilerDesc(frame_stats_window_size=2))
    with spy.profile_frame():
        for _ in range(4):
            with spy.profile_zone("dispatch"):
                pass
    with spy.profile_frame():
        pass
    profiler.flush()

    stats = profiler.frame_stats_snapshot()
    assert stats.sample_count == 2
    assert len(stats.entries) == 1
    dispatch = stats.entries[0]
    assert dispatch.cpu_time_per_frame.count == 2
    assert dispatch.cpu_time_per_call.count == 4
    assert dispatch.gpu_time_per_frame.count == 0
    assert stats.sample_frame_index.shape == (2,)
    assert stats.sample_frame_time_ms.shape == (2,)
    assert stats.sample_call_count.shape == (2, 1)
    assert stats.sample_cpu_time_ms.shape == (2, 1)
    assert stats.sample_gpu_time_ms.shape == (2, 1)
    assert stats.sample_gpu_status.shape == (2, 1)
    assert stats.sample_frame_index[0] < stats.sample_frame_index[1]
    assert stats.sample_call_count.tolist() == [[4], [0]]
    assert stats.sample_cpu_time_ms[0, 0] >= 0.0
    assert stats.sample_cpu_time_ms[1, 0] == 0.0
    assert stats.sample_gpu_time_ms[:, 0].tolist() == [0.0, 0.0]
    assert stats.sample_gpu_status[0, 0] == spy.ProfilerFrameGpuStatus.unavailable.value
    assert stats.sample_gpu_status[1, 0] == spy.ProfilerFrameGpuStatus.absent.value
    assert not stats.sample_call_count.flags.writeable
    assert not stats.sample_gpu_status.flags.writeable


def test_empty_frame_statistics_have_well_formed_matrices() -> None:
    profiler = spy.Profiler()
    with spy.profile_frame("empty"):
        pass
    profiler.flush()

    stats = profiler.frame_stats_snapshot()
    assert stats.sample_count == 1
    assert stats.entry_count == 0
    assert stats.sample_frame_index.shape == (1,)
    assert stats.sample_call_count.shape == (1, 0)
    assert stats.sample_cpu_time_ms.shape == (1, 0)
    assert stats.sample_gpu_time_ms.shape == (1, 0)
    assert stats.sample_gpu_status.shape == (1, 0)


def test_capture_state_errors_and_memory_limit() -> None:
    profiler = spy.Profiler()
    with pytest.raises(RuntimeError, match="No active"):
        profiler.stop_capture()
    with pytest.raises(RuntimeError, match="No active"):
        profiler.discard_capture()

    profiler.start_capture(spy.ProfilerCaptureDesc(max_memory_bytes=64))
    with pytest.raises(RuntimeError, match="already active"):
        profiler.start_capture()
    for _ in range(32):
        with spy.profile_zone("limited"):
            pass
    profiler.flush()
    assert not profiler.capture_active
    trace = profiler.stop_capture()
    assert trace.truncated
    assert trace.stop_reason == spy.ProfilerCaptureStopReason.memory_limit
    assert trace.memory_bytes <= 64


def test_disabled_profiler_does_not_record() -> None:
    profiler = spy.Profiler()
    profiler.enabled = False
    profiler.start_capture()
    with spy.profile_zone("disabled"):
        pass
    assert profiler.stop_capture().zone_count == 0


def test_nested_frames_are_rejected() -> None:
    profiler = spy.Profiler()
    with spy.profile_frame():
        with pytest.raises(RuntimeError, match="already active"):
            with spy.profile_frame("nested"):
                pass


def test_json_export_and_array_ownership(tmp_path: Path) -> None:
    profiler = spy.Profiler()
    profiler.start_capture()
    with spy.profile_zone('json "escape"'):
        pass
    trace = profiler.stop_capture()
    values = trace.zone_chunks[0].duration_ns
    output = tmp_path / "trace.json"
    trace.write_to_json(output)
    text = output.read_text(encoding="utf-8")
    assert 'json \\"escape\\"' in text
    assert '"ph":"X"' in text

    del trace
    del profiler
    assert values.shape == (1,)
    assert int(values[0]) >= 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_manual_gpu_zone_and_automatic_functional_zone(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    profiler = spy.Profiler()
    profiler.start_capture()

    command_encoder = device.create_command_encoder()
    with spy.profile_zone("manual gpu", command_encoder):
        pass
    device.submit_command_buffer(command_encoder.finish())

    module = spy.Module(
        device.load_module_from_source(
            "test_profiler_functional",
            "int add(int a, int b) { return a + b; }",
        )
    )
    assert module.add(1, 2) == 3

    device.wait()
    profiler.tick()
    trace = profiler.stop_capture()
    assert (
        trace.query_zones(name="manual gpu", timeline_type=spy.ProfilerTimelineType.cpu).count == 1
    )
    assert (
        trace.query_zones(
            name="test_profiler_functional::add",
            timeline_type=spy.ProfilerTimelineType.cpu,
        ).count
        == 1
    )
    if device.has_feature(spy.Feature.timestamp_query) and device.has_feature(
        spy.Feature.timestamp_calibration
    ):
        assert (
            trace.query_zones(name="manual gpu", timeline_type=spy.ProfilerTimelineType.gpu).count
            == 1
        )
        assert trace.diagnostics.pending_gpu_zone_count == 0
