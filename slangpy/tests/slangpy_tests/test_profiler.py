# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import sys
import textwrap
import threading
import time
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
    assert chunk.frame_index.dtype == np.uint32
    assert np.all(chunk.frame_index != np.iinfo(np.uint32).max)
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
    assert stats.sample_frame_start_ns.shape == (1,)
    assert stats.sample_frame_interval_ms.tolist() == [0.0]
    assert stats.frame_interval.count == 0


def test_frame_intervals_measure_cadence_and_survive_window_eviction() -> None:
    profiler = spy.Profiler(spy.ProfilerDesc(frame_stats_window_size=2))
    profiler.start_capture()
    for _ in range(3):
        with spy.profile_frame():
            pass
        time.sleep(0.02)
    trace = profiler.stop_capture()
    stats = profiler.frame_stats_snapshot()
    frames = trace.frames
    assert len(frames) == 3
    assert frames[0].interval_ns == 0
    for previous, current in zip(frames, frames[1:]):
        assert current.interval_ns == current.start_ns - previous.start_ns
        # The deliberate gap is outside the CPU scope, but inside the interval.
        assert current.interval_ns - previous.duration_ns >= 15_000_000
    expected_intervals = np.array([frame.interval_ns * 1e-6 for frame in frames[1:]])
    np.testing.assert_allclose(stats.sample_frame_interval_ms, expected_intervals)
    np.testing.assert_array_equal(
        stats.sample_frame_start_ns, [frame.start_ns for frame in frames[1:]]
    )
    np.testing.assert_allclose(
        stats.sample_frame_time_ms, [frame.duration_ns * 1e-6 for frame in frames[1:]]
    )
    assert stats.latest_frame_ms == stats.sample_frame_time_ms[-1]
    assert stats.frame_interval.count == 2
    assert stats.frame_interval.mean_ms == pytest.approx(np.mean(expected_intervals))
    assert stats.frame_interval.p95_ms == pytest.approx(np.percentile(expected_intervals, 95))
    assert stats.sample_frame_start_ns.dtype == np.uint64
    assert stats.sample_frame_interval_ms.dtype == np.float64
    starts = stats.sample_frame_start_ns
    intervals = stats.sample_frame_interval_ms
    assert not starts.flags.writeable
    assert not intervals.flags.writeable
    profiler.clear_frame_stats()
    cleared = profiler.frame_stats_snapshot()
    assert cleared.frame_interval.count == 0
    assert cleared.sample_frame_start_ns.shape == (0,)
    assert cleared.sample_frame_interval_ms.shape == (0,)
    del cleared, stats, trace, profiler
    np.testing.assert_allclose(intervals, expected_intervals)
    assert starts.shape == (2,)


def test_frame_intervals_restart_after_disabled_recording() -> None:
    profiler = spy.Profiler()
    with spy.profile_frame():
        pass
    profiler.enabled = False
    with spy.profile_frame():
        pass
    profiler.enabled = True
    with spy.profile_frame():
        pass
    profiler.enabled = True  # An unchanged setting must not reset cadence.
    with spy.profile_frame():
        pass
    profiler.flush()
    stats = profiler.frame_stats_snapshot()
    assert stats.sample_frame_interval_ms[:2].tolist() == [0.0, 0.0]
    assert stats.sample_frame_interval_ms[2] > 0.0
    assert stats.frame_interval.count == 1
    assert stats.frame_interval.mean_ms == stats.sample_frame_interval_ms[2]


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
    with spy.profile_frame("disabled frame"):
        with spy.profile_zone("disabled zone"):
            with spy.profile_function():
                pass
    trace = profiler.stop_capture()
    assert trace.zone_count == 0
    assert all(site.name not in {"disabled frame", "disabled zone"} for site in trace.sites)
    assert all(
        not site.name.endswith("test_disabled_profiler_does_not_record") for site in trace.sites
    )


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
    assert trace.zone_chunks[0].frame_index[0] == np.iinfo(np.uint32).max
    assert trace.query_zones(frame_begin=0).count == 0
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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("separate_encoder", [False, True])
def test_gpu_child_history_is_visible_without_gpu_parent(
    device_type: spy.DeviceType, separate_encoder: bool
) -> None:
    device = helpers.get_device(device_type)
    required = (
        spy.Feature.timestamp_query,
        spy.Feature.timestamp_calibration,
        spy.Feature.rasterization,
    )
    if not all(device.has_feature(feature) for feature in required):
        pytest.skip("Requires GPU timestamps and UI rasterization")
    profiler = spy.Profiler()
    buffer = device.create_buffer(size=4 * 1024 * 1024, usage=spy.BufferUsage.unordered_access)
    for _ in range(3):
        with spy.profile_frame():
            encoder = device.create_command_encoder()
            child_encoder = device.create_command_encoder() if separate_encoder else encoder
            with spy.profile_zone("CPU-only parent"):
                with spy.profile_zone("GPU child", encoder):
                    for _ in range(4):
                        encoder.clear_buffer(buffer)
                    with spy.profile_zone("GPU grandchild", child_encoder):
                        for _ in range(256 if separate_encoder else 4):
                            child_encoder.clear_buffer(buffer)
            if separate_encoder:
                device.submit_command_buffer(child_encoder.finish())
            device.submit_command_buffer(encoder.finish())
    device.wait()
    deadline = time.monotonic() + 10.0
    while True:
        profiler.tick()
        profiler.flush()
        stats = profiler.frame_stats_snapshot()
        if stats.diagnostics.pending_gpu_zone_count == 0:
            break
        assert time.monotonic() < deadline, "GPU profiler queries did not resolve"
        time.sleep(0.001)
    assert stats.entries[0].gpu_time_per_call.count == 0
    assert stats.entries[1].parent_index == 0
    assert stats.entries[1].gpu_time_per_frame.maximum_ms > 0.0
    assert stats.entries[2].parent_index == 1
    assert stats.entries[2].gpu_time_per_frame.maximum_ms > 0.0

    context = spy.ui.Context(device)
    target = device.create_texture(
        format=spy.Format.rgba8_unorm,
        width=1024,
        height=900,
        usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
    )
    # ImGui needs a layout frame before newly created windows become visible.
    for _ in range(3):
        encoder = device.create_command_encoder()
        encoder.clear_texture_float(target, clear_value=spy.float4(0, 0, 0, 1))
        context.begin_frame(1024, 900)
        spy.ui.render_profiler_window(profiler)
        context.end_frame(target, encoder)
        device.submit_command_buffer(encoder.finish())
    pixels = target.to_numpy().reshape(900, 1024, 4)
    # A single plot must show a contiguous stack beneath the selected CPU-only root,
    # rather than separate lanes or children clipped to the unmeasured parent.
    # Exclude the blue mode selector and axis labels above/beside the plot.
    history = pixels[510:775, 180:990, :3].astype(np.int16)
    colored = history.max(axis=2) - history.min(axis=2) > 60
    assert np.count_nonzero(colored) > 5000
    column = int(np.argmax(np.count_nonzero(colored, axis=0)))
    rows = np.flatnonzero(colored[:, column])
    assert len(rows) > 200  # The stack uses the full height of one plot.
    assert np.all(np.diff(rows) == 1)
    colors = history[rows, column]
    transitions = np.flatnonzero(np.any(colors[1:] != colors[:-1], axis=1))
    child = stats.sample_gpu_time_ms[:, 1]
    grandchild = stats.sample_gpu_time_ms[:, 2]
    if separate_encoder:
        # CPU nesting does not imply GPU containment across command encoders.
        # The larger grandchild must survive without being clipped or rescaled.
        assert np.all(grandchild > child)
        assert len(transitions) == 0
    else:
        # Only the parent's remainder is added to the nested grandchild's band.
        assert len(transitions) == 1
        largest = int(np.argmax(child))
        grandchild_fraction = float(grandchild[largest] / child[largest])
        assert (transitions[0] + 1) / len(rows) == pytest.approx(grandchild_fraction, abs=0.03)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_recent_frames_do_not_wait_for_gpu(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    if not all(
        device.has_feature(f)
        for f in (spy.Feature.timestamp_query, spy.Feature.timestamp_calibration)
    ):
        pytest.skip("Requires GPU timestamps")
    profiler = spy.Profiler(spy.ProfilerDesc(frame_stats_window_size=4))
    assert profiler.frame_stats_snapshot(include_pending=True).sample_count == 0
    with spy.profile_frame():
        with spy.profile_zone("baseline"):
            pass
    profiler.flush()
    baseline = profiler.frame_stats_snapshot()
    assert baseline.sample_frame_index.tolist() == [0]

    def record_gpu(calls: int) -> spy.CommandBuffer:
        encoder = device.create_command_encoder()
        with spy.profile_frame():
            for _ in range(calls):
                with spy.profile_zone("delayed", encoder):
                    pass
        return encoder.finish()

    first = record_gpu(1)
    second = record_gpu(2)
    with spy.profile_frame():
        with spy.profile_zone("new CPU zone"):
            pass
    profiler.flush()
    recent = profiler.frame_stats_snapshot(include_pending=True)
    completed = profiler.frame_stats_snapshot()
    assert completed.sample_frame_index.tolist() == [0]
    assert [entry.name for entry in completed.entries] == ["baseline"]
    assert recent.sample_frame_index.tolist() == [0, 1, 2, 3]
    assert recent.frame_time.count == 4
    assert recent.frame_interval.count == 3
    assert recent.pending_frame_count == 3
    delayed = next(i for i, entry in enumerate(recent.entries) if entry.name == "delayed")
    cpu_zone = next(i for i, entry in enumerate(recent.entries) if entry.name == "new CPU zone")
    assert recent.sample_call_count[:, delayed].tolist() == [0, 1, 2, 0]
    assert recent.sample_call_count[:, cpu_zone].tolist() == [0, 0, 0, 1]
    pending = spy.ProfilerFrameGpuStatus.pending.value
    complete = spy.ProfilerFrameGpuStatus.complete.value
    absent = spy.ProfilerFrameGpuStatus.absent.value
    assert recent.sample_gpu_status[:, delayed].tolist() == [absent, pending, pending, absent]
    assert recent.entries[delayed].gpu_time_per_frame.count == 0
    statuses = recent.sample_gpu_status
    frame_ids = recent.sample_frame_index
    assert not statuses.flags.writeable
    assert not frame_ids.flags.writeable

    # Resolve frame 2 first. Its measurement must not migrate into frame 1 or 3,
    # and the completed table must still wait for frame 1.
    device.submit_command_buffer(second)
    device.wait()
    deadline = time.monotonic() + 10.0
    while True:
        profiler.tick()
        profiler.flush()
        out_of_order = profiler.frame_stats_snapshot(include_pending=True)
        if out_of_order.sample_gpu_status[2, delayed] == complete:
            break
        assert time.monotonic() < deadline, "GPU queries did not resolve"
        time.sleep(0.001)
    assert out_of_order.sample_gpu_status[:, delayed].tolist() == [
        absent,
        pending,
        complete,
        absent,
    ]
    assert out_of_order.entries[delayed].gpu_time_per_call.count == 2
    assert profiler.frame_stats_snapshot().sample_frame_index.tolist() == [0]

    device.submit_command_buffer(first)
    device.wait()
    while True:
        profiler.tick()
        profiler.flush()
        resolved = profiler.frame_stats_snapshot()
        if resolved.pending_frame_count == 0:
            break
        assert time.monotonic() < deadline, "GPU queries did not resolve"
        time.sleep(0.001)
    assert resolved.sample_frame_index.tolist() == [0, 1, 2, 3]
    assert resolved.sample_gpu_status[:, delayed].tolist() == [absent, complete, complete, absent]
    assert resolved.entries[delayed].gpu_time_per_call.count == 3
    np.testing.assert_array_equal(resolved.sample_cpu_time_ms, recent.sample_cpu_time_ms)
    np.testing.assert_array_equal(resolved.sample_frame_start_ns, recent.sample_frame_start_ns)
    np.testing.assert_array_equal(
        resolved.sample_frame_interval_ms, recent.sample_frame_interval_ms
    )
    # Publishing resolved results never mutates earlier Python views.
    assert statuses[:, delayed].tolist() == [absent, pending, pending, absent]

    # Both windows are bounded, and clearing rejects late results from old frames.
    third = record_gpu(1)
    profiler.flush()
    assert profiler.frame_stats_snapshot(include_pending=True).sample_frame_index.tolist() == [
        1,
        2,
        3,
        4,
    ]
    assert profiler.frame_stats_snapshot().sample_frame_index.tolist() == [0, 1, 2, 3]
    profiler.clear_frame_stats()
    assert profiler.frame_stats_snapshot(include_pending=True).sample_count == 0
    device.submit_command_buffer(third)
    device.wait()
    profiler.tick()
    profiler.flush()
    assert profiler.frame_stats_snapshot().sample_count == 0
    assert profiler.frame_stats_snapshot(include_pending=True).sample_count == 0
    del recent, completed, baseline, resolved, out_of_order, profiler
    assert frame_ids.tolist() == [0, 1, 2, 3]
    assert statuses[:, delayed].tolist() == [absent, pending, pending, absent]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_profiler_window_releases_python_snapshots_at_exit(device_type: spy.DeviceType) -> None:
    if not helpers.get_device(device_type).has_feature(spy.Feature.rasterization):
        pytest.skip("Requires UI rasterization")
    # This must run in a subprocess: retaining a Python-owned snapshot in the
    # native window cache previously crashed during interpreter finalization.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
            import slangpy as spy
            device = spy.Device(type=spy.DeviceType.{device_type.name})
            profiler = spy.Profiler()
            with spy.profile_frame():
                with spy.profile_zone("CPU scope"):
                    pass
            profiler.flush()
            context = spy.ui.Context(device)
            target = device.create_texture(format=spy.Format.rgba8_unorm, width=1024, height=900,
                usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource)
            for _ in range(3):
                encoder = device.create_command_encoder()
                context.begin_frame(1024, 900)
                spy.ui.render_profiler_window(profiler)
                context.end_frame(target, encoder)
                device.submit_command_buffer(encoder.finish())
            # Convert the already-cached native snapshot to Python ownership.
            recent = profiler.frame_stats_snapshot(include_pending=True)
            assert recent.sample_count == 1
            device.wait()
            device.close()
        """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
