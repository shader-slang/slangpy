# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Device-wide collection through functional calls and explicit compute programs."""

import gc
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import weakref
from typing import Iterator

import numpy as np
import pytest
import slangpy as spy
from slangpy.testing import helpers

pytestmark = pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)


@pytest.fixture(autouse=True)
def require_coverage_backend(device_type: spy.DeviceType) -> None:
    if device_type not in (spy.DeviceType.vulkan, spy.DeviceType.cuda):
        pytest.skip("Shader coverage currently supports Vulkan and CUDA compute programs")


SOURCE = Path(__file__).with_name("test_shader_coverage_identity.slang")


@pytest.fixture
def device(device_type: spy.DeviceType) -> Iterator[spy.Device]:
    device = spy.Device(
        type=device_type,
        enable_debug_layers=True,
        compiler_options={"coverage": spy.ShaderCoverageOptions(counter_width=32)},
    )
    yield device
    device.close()


def arrays(device: spy.Device) -> tuple[spy.Tensor, spy.Tensor]:
    values = np.array([-3, -1, 0, 1, 3], dtype=np.int32)
    return spy.Tensor.from_numpy(device, values), spy.Tensor.from_numpy(
        device, np.zeros_like(values)
    )


def branch_counts(program: spy.ShaderCoverageProgramSnapshot) -> list[int]:
    manifest = json.loads(program.manifest)
    assert len(program.counters) == manifest["counter_count"]
    return [
        program.counters[entry["counter"]]
        for entry in manifest["entries"]
        if entry["kind"] == "branch" and Path(entry["file"]).name == SOURCE.name
    ]


def test_collects_multiple_programs_and_keeps_macro_arms(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    collector = device.shader_coverage
    assert collector.capabilities.supported
    assert 32 in collector.capabilities.counter_widths
    assert (64 in collector.capabilities.counter_widths) == device.has_feature(
        spy.Feature.atomic_int64
    )
    assert collector.snapshot().programs == []
    module = spy.Module.load_from_file(device, str(SOURCE))
    inputs, output = arrays(device)
    module.wrapperA(inputs, _result=output)
    module.repeatedMacro(inputs, _result=output)
    snapshot = collector.snapshot()
    assert len(snapshot.programs) == 2
    assert len({p.generation_id for p in snapshot.programs}) == 2
    assert sorted(branch_counts(snapshot.programs[0])) == [2, 3]
    assert sorted(branch_counts(snapshot.programs[1])) == [0, 2, 3, 5]
    assert sum(v > 0 for v in branch_counts(snapshot.programs[1])) == 3
    np.testing.assert_array_equal(output.to_numpy(), [3, 3, 4, 4, 4])

    # Registry ownership survives clearing generated pipelines and Python module references.
    del module
    gc.collect()
    retained = collector.snapshot()
    assert [p.counters for p in retained.programs] == [p.counters for p in snapshot.programs]


def test_deferred_and_snapshot_reset(device_type: spy.DeviceType, device: spy.Device) -> None:
    collector = device.shader_coverage
    module = spy.Module.load_from_file(device, str(SOURCE))
    inputs, output = arrays(device)
    encoder = device.create_command_encoder()
    module.wrapperA.append_to(encoder, inputs, _result=output)
    commands = encoder.finish()
    pending = collector.snapshot(reset=True)
    assert len(pending.programs) == 1
    assert not any(pending.programs[0].counters)
    device.submit_command_buffer(commands)
    first = collector.snapshot(reset=True)
    assert sorted(branch_counts(first.programs[0])) == [2, 3]
    assert first.interval_id == pending.interval_id + 1
    assert first.reset_after
    assert not any(collector.snapshot().programs[0].counters)
    # RHI command buffers are single-use. Record fresh calls for the next interval.
    module.wrapperA(inputs, _result=output)
    module.wrapperA(inputs, _result=output)
    twice = collector.snapshot()
    assert sorted(branch_counts(twice.programs[0])) == [4, 6]
    assert twice.interval_id == first.interval_id + 1
    assert twice.collection_id == first.collection_id
    assert twice.capture_id > first.capture_id
    collector.reset()
    assert not any(collector.snapshot().programs[0].counters)


def test_snapshot_survives_close_and_is_read_only(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    module = spy.Module.load_from_file(device, str(SOURCE))
    inputs, output = arrays(device)
    module.wrapperA(inputs, _result=output)
    collector = device.shader_coverage
    snapshot = collector.snapshot()
    before = snapshot.programs[0].counters
    with pytest.raises(AttributeError):
        snapshot.interval_id = 99
    with pytest.raises(AttributeError):
        snapshot.programs[0].manifest = "{}"
    # Returned containers are copies: edits cannot mutate captured native data.
    snapshot.programs[0].counters[0] = 999999
    snapshot.programs.clear()
    device.close()
    assert snapshot.programs[0].counters == before
    assert sorted(branch_counts(snapshot.programs[0])) == [2, 3]
    with pytest.raises(RuntimeError, match="Device is closed"):
        collector.snapshot()


def test_explicit_compute_and_second_session(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    session = device.create_slang_session(
        compiler_options={"coverage": {"counter_width": 32, "lines": False, "branches": False}}
    )
    source = session.load_module_from_source(
        "coverage_explicit",
        '[shader("compute")][numthreads(1,1,1)] void computeMain(uint3 tid : SV_DispatchThreadID) {}',
    )
    program = session.link_program([source], [source.entry_point("computeMain")])
    kernel = device.create_compute_kernel(program)
    assert len(device.shader_coverage.snapshot().programs) == 1
    kernel.dispatch(thread_count=[7, 1, 1])
    snapshot = device.shader_coverage.snapshot()
    assert len(snapshot.programs) == 1
    record = snapshot.programs[0]
    manifest = json.loads(record.manifest)
    assert {e["kind"] for e in manifest["entries"]} == {"function"}
    assert record.counters == [7]


def test_disabled_and_invalid_modes(device_type: spy.DeviceType) -> None:
    assert spy.SlangCompilerOptions().coverage is None
    assert spy.ShaderCoverageOptions().counter_width == 64
    options = spy.SlangCompilerOptions()
    options.coverage = {"counter_width": 32}
    assert options.coverage.counter_width == 32
    options.coverage = None
    assert options.coverage is None
    with spy.Device(type=device_type) as device:
        module = spy.Module.load_from_file(device, str(SOURCE))
        inputs, output = arrays(device)
        module.wrapperA(inputs, _result=output)
        assert device.shader_coverage.snapshot().programs == []
        device.shader_coverage.reset()
    with spy.Device(
        type=device_type,
        compiler_options={
            "coverage": {"counter_width": 32, "lines": False, "functions": False, "branches": False}
        },
    ) as device:
        module = device.load_module_from_source(
            "no_modes", '[shader("compute")][numthreads(1,1,1)] void computeMain() {}'
        )
        with pytest.raises(RuntimeError, match="At least one coverage mode"):
            device.slang_session.link_program([module], [module.entry_point("computeMain")])


@pytest.mark.parametrize("collect_empty_first", [False, True])
def test_enable_coverage_after_ordinary_dispatch(
    device_type: spy.DeviceType, collect_empty_first: bool
) -> None:
    """A later instrumented session must not alter existing ordinary programs."""
    source = """
RWStructuredBuffer<uint> output;
[shader("compute")][numthreads(1, 1, 1)]
void compute_main() { output[0] = 42; }
"""
    with spy.Device(type=device_type) as device:
        ordinary_module = device.load_module_from_source("ordinary_then_covered", source)
        ordinary = device.slang_session.link_program(
            [ordinary_module], [ordinary_module.entry_point("compute_main")]
        )
        output = device.create_buffer(
            element_count=1,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access,
        )
        ordinary_kernel = device.create_compute_kernel(ordinary)
        ordinary_kernel.dispatch(thread_count=[1, 1, 1], vars={"output": output})
        assert ordinary.coverage_buffer is None
        assert ordinary.coverage_manifest == ""
        if collect_empty_first:
            assert device.shader_coverage.snapshot(reset=True).programs == []

        session = device.create_slang_session(compiler_options={"coverage": {"counter_width": 32}})
        covered_module = session.load_module_from_source("covered_later", source)
        covered = session.link_program(
            [covered_module], [covered_module.entry_point("compute_main")]
        )
        device.create_compute_kernel(covered).dispatch(
            thread_count=[1, 1, 1], vars={"output": output}
        )
        first = device.shader_coverage.snapshot()
        assert len(first.programs) == 1
        assert any(first.programs[0].counters)

        ordinary_kernel.dispatch(thread_count=[1, 1, 1], vars={"output": output})
        second = device.shader_coverage.snapshot()
        assert second.programs[0].counters == first.programs[0].counters
        assert ordinary.coverage_buffer is None
        assert ordinary.coverage_manifest == ""
        np.testing.assert_array_equal(output.to_numpy().view(np.uint32), [42])


def test_device_registry_does_not_keep_closed_device_alive(device_type: spy.DeviceType) -> None:
    device = spy.Device(type=device_type, compiler_options={"coverage": {"counter_width": 32}})
    module = spy.Module.load_from_file(device, str(SOURCE))
    inputs, output = arrays(device)
    module.wrapperA(inputs, _result=output)
    snapshot = device.shader_coverage.snapshot()
    reference = weakref.ref(device)
    device.close()
    del module, inputs, output, device
    gc.collect()
    assert reference() is None
    assert sorted(branch_counts(snapshot.programs[0])) == [2, 3]


def test_reload_preserves_pending_generation(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    module = spy.Module.load_from_file(device, str(SOURCE))
    inputs, output = arrays(device)
    encoder = device.create_command_encoder()
    module.wrapperA.append_to(encoder, inputs, _result=output)
    pending = encoder.finish()
    before = device.shader_coverage.snapshot()
    old_id = before.programs[0].generation_id
    device.reload_all_programs()
    reloaded = device.shader_coverage.snapshot()
    assert old_id in {p.generation_id for p in reloaded.programs}
    assert len(reloaded.programs) > len(before.programs)
    assert all(not any(p.counters) for p in reloaded.programs)
    device.submit_command_buffer(pending)
    executed = device.shader_coverage.snapshot()
    old = next(p for p in executed.programs if p.generation_id == old_id)
    assert sorted(branch_counts(old)) == [2, 3]
    assert all(not any(p.counters) for p in executed.programs if p.generation_id != old_id)
    module.wrapperA(inputs, _result=output)
    latest = device.shader_coverage.snapshot()
    assert sum(sum(branch_counts(p)) for p in latest.programs) == 10
    assert sorted(branch_counts(next(p for p in latest.programs if p.generation_id == old_id))) == [
        2,
        3,
    ]


def test_internal_collection_does_not_notify_user_submission_callbacks(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    events: list[int] = []
    callback = device.register_command_recording_submitted_callback(
        lambda event: events.append(event.submit_id)
    )
    try:
        module = spy.Module.load_from_file(device, str(SOURCE))
        inputs, output = arrays(device)
        module.wrapperA(inputs, _result=output)
        before = len(events)
        device.shader_coverage.snapshot(reset=True)
        device.shader_coverage.reset()
        assert len(events) == before
    finally:
        device.unregister_command_recording_submitted_callback(callback)


@pytest.mark.parametrize(
    "modes",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ],
)
def test_instrumentation_modes(
    device_type: spy.DeviceType, device: spy.Device, modes: tuple[bool, bool, bool]
) -> None:
    lines, functions, branches = modes
    session = device.create_slang_session(
        compiler_options={
            "coverage": {
                "counter_width": 32,
                "lines": lines,
                "functions": functions,
                "branches": branches,
            }
        }
    )
    source = session.load_module_from_source(
        "coverage_modes",
        'RWStructuredBuffer<int> output; [shader("compute")][numthreads(1,1,1)] void computeMain(uint3 tid : SV_DispatchThreadID) { if (tid.x < 2) output[tid.x] = 10; else output[tid.x] = 20; }',
    )
    program = session.link_program([source], [source.entry_point("computeMain")])
    output = device.create_buffer(size=16, usage=spy.BufferUsage.unordered_access)
    device.create_compute_kernel(program).dispatch(thread_count=[4, 1, 1], vars={"output": output})
    snapshot = device.shader_coverage.snapshot()
    assert len(snapshot.programs) == 1
    record = snapshot.programs[0]
    kinds = {e["kind"] for e in json.loads(record.manifest)["entries"]}
    assert kinds == {
        name for name, enabled in zip(("line", "function", "branch"), modes) if enabled
    }
    assert any(record.counters)


def test_submission_callback_can_capture(device_type: spy.DeviceType, device: spy.Device) -> None:
    module = spy.Module.load_from_file(device, str(SOURCE))
    inputs, output = arrays(device)
    encoder = device.create_command_encoder()
    module.wrapperA.append_to(encoder, inputs, _result=output)
    commands = encoder.finish()
    captures: list[spy.ShaderCoverageSnapshot] = []
    callback = device.register_command_recording_submitted_callback(
        lambda event: captures.append(device.shader_coverage.snapshot())
    )
    try:
        device.submit_command_buffer(commands)
        assert len(captures) == 1
        assert sorted(branch_counts(captures[0].programs[0])) == [2, 3]
    finally:
        device.unregister_command_recording_submitted_callback(callback)


def test_no_matching_instrumentation_sites(device_type: spy.DeviceType, device: spy.Device) -> None:
    session = device.create_slang_session(
        compiler_options={"coverage": {"counter_width": 32, "lines": False, "functions": False}}
    )
    module = session.load_module_from_source(
        "coverage_empty", '[shader("compute")][numthreads(1,1,1)] void computeMain() {}'
    )
    program = session.link_program([module], [module.entry_point("computeMain")])
    device.create_compute_kernel(program).dispatch(thread_count=[1, 1, 1])
    snapshot = device.shader_coverage.snapshot(reset=True)
    assert len(snapshot.programs) == 1
    assert snapshot.programs[0].counters == []
    assert json.loads(snapshot.programs[0].manifest)["entries"] == []


def test_failed_reload_keeps_previous_counts(
    device_type: spy.DeviceType, device: spy.Device, tmp_path: Path
) -> None:
    source = tmp_path / "coverage_failed_reload.slang"
    source.write_text('[shader("compute")][numthreads(1,1,1)] void computeMain() {}')
    program = device.load_program(str(source), ["computeMain"])
    kernel = device.create_compute_kernel(program)
    kernel.dispatch(thread_count=[3, 1, 1])
    before = device.shader_coverage.snapshot()
    source.write_text("this is not valid Slang;")
    # Existing hot-reload API logs compilation failure and preserves the old program.
    device.reload_all_programs()
    after = device.shader_coverage.snapshot()
    assert [p.generation_id for p in after.programs] == [p.generation_id for p in before.programs]
    assert [p.counters for p in after.programs] == [p.counters for p in before.programs]


def test_collection_releases_gil(device_type: spy.DeviceType) -> None:
    """A Python thread can unblock the GPU while snapshot or reset waits for it."""
    if device_type != spy.DeviceType.vulkan:
        pytest.skip("This test uses Vulkan's asynchronous queue wait on a host-signaled fence")
    # Isolate a potential GIL deadlock so a regression fails instead of hanging pytest.
    script = textwrap.dedent(
        """
        import threading
        import slangpy as spy

        with spy.Device(
            type=spy.DeviceType.vulkan,
            compiler_options={"coverage": {"counter_width": 32}},
        ) as device:
            module = device.load_module_from_source(
                "coverage_gil",
                '[shader("compute")][numthreads(1,1,1)] void compute_main() {}',
            )
            program = device.slang_session.link_program(
                [module], [module.entry_point("compute_main")]
            )
            kernel = device.create_compute_kernel(program)
            kernel.dispatch(thread_count=[1, 1, 1])
            collector = device.shader_coverage
            assert any(collector.snapshot().programs[0].counters)

            for operation in (collector.snapshot, lambda: collector.snapshot(reset=True), collector.reset):
                gate = device.create_fence()
                commands = device.create_command_encoder().finish()
                device.submit_command_buffers(
                    [commands], wait_fences=[gate], wait_fence_values=[1]
                )
                # The queue cannot complete collection until this Python timer runs.
                signal = threading.Timer(0.2, gate.signal, args=(1,))
                signal.start()
                try:
                    operation()
                finally:
                    signal.join()
                assert gate.current_value == 1
            assert not any(collector.snapshot().programs[0].counters)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
