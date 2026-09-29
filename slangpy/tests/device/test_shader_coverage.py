# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Coverage instrumentation, counter widths, and automatic resource binding."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import slangpy as spy
from slangpy.testing import helpers

pytestmark = pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)


@pytest.fixture(autouse=True)
def require_coverage_backend(device_type: spy.DeviceType) -> None:
    if device_type not in (spy.DeviceType.vulkan, spy.DeviceType.cuda):
        pytest.skip("Shader coverage currently supports Vulkan and CUDA compute programs")


SOURCE = Path(__file__).with_suffix(".slang")


def create_device(device_type: spy.DeviceType, width: int = 32, enabled: bool = True) -> spy.Device:
    return spy.Device(
        type=device_type,
        enable_debug_layers=True,
        compiler_options={
            "coverage": {"counter_width": width} if enabled else None,
        },
    )


def read_coverage(program: spy.ShaderProgram) -> tuple[dict[str, Any], np.ndarray]:
    manifest = json.loads(program.coverage_manifest)
    stride = manifest["buffer"]["element_stride"]
    counters = program.coverage_buffer.to_numpy().view(f"<u{stride}").reshape(-1)
    assert counters.size == manifest["counter_count"]
    return manifest, counters


def assert_source_counts(manifest: dict[str, Any], counters: np.ndarray, calls: int) -> None:
    entries = [e for e in manifest["entries"] if Path(e["file"]).name == SOURCE.name]
    assert entries
    branches = [e for e in entries if e["kind"] == "branch"]
    assert len(branches) == 2
    assert sorted(int(counters[e["counter"]]) for e in branches) == [2 * calls, 2 * calls]
    functions = [e for e in entries if e["kind"] == "function"]
    assert len(functions) == 1
    assert int(counters[functions[0]["counter"]]) == 4 * calls
    source_lines = SOURCE.read_text().splitlines()
    for needle, expected in (("if (x < 0)", 4), ("return -x;", 2), ("return 2 * x;", 2)):
        line = next(i + 1 for i, s in enumerate(source_lines) if needle in s)
        hits = [
            int(counters[e["counter"]])
            for e in entries
            if e["kind"] == "line" and e["line"] == line
        ]
        assert hits == [expected * calls]


def test_functional_coverage_accumulation_deferred_and_reset(device_type: spy.DeviceType) -> None:
    device = create_device(device_type)
    module = spy.Module.load_from_file(device, str(SOURCE))
    inputs = spy.Tensor.from_numpy(device, np.array([-3, -1, 0, 2], dtype=np.int32))
    # Automatic output allocation requests shared memory unsupported by MoltenVK.
    output = spy.Tensor.from_numpy(device, np.zeros(4, dtype=np.int32))
    module.classify(inputs, _result=output)
    np.testing.assert_array_equal(output.to_numpy(), [3, 1, 0, 4])
    program = next(iter(module.pipeline_cache.values())).program
    manifest, first = read_coverage(program)
    assert manifest["buffer"]["element_stride"] == 4
    assert_source_counts(manifest, first, 1)

    module.classify(inputs, _result=output)
    _, second = read_coverage(program)
    assert len(module.pipeline_cache) == 1
    np.testing.assert_array_equal(second, 2 * first)
    assert_source_counts(manifest, second, 2)

    encoder = device.create_command_encoder()
    module.classify.append_to(encoder, inputs, _result=output)
    module.classify.append_to(encoder, inputs, _result=output)
    device.submit_command_buffer(encoder.finish())
    _, fourth = read_coverage(program)
    np.testing.assert_array_equal(fourth, 4 * first)
    assert_source_counts(manifest, fourth, 4)

    # Readback above waits for preceding execution. Reset only after completion.
    program.coverage_buffer.copy_from_numpy(np.zeros_like(first))
    module.classify(inputs, _result=output)
    _, reset = read_coverage(program)
    np.testing.assert_array_equal(reset, first)


def test_disabled_coverage_preserves_results(device_type: spy.DeviceType) -> None:
    device = create_device(device_type, enabled=False)
    module = spy.Module.load_from_file(device, str(SOURCE))
    inputs = spy.Tensor.from_numpy(device, np.array([-3, -1, 0, 2], dtype=np.int32))
    output = spy.Tensor.from_numpy(device, np.zeros(4, dtype=np.int32))
    module.classify(inputs, _result=output)
    np.testing.assert_array_equal(output.to_numpy(), [3, 1, 0, 4])
    program = next(iter(module.pipeline_cache.values())).program
    assert program.coverage_buffer is None
    assert program.coverage_manifest == ""


def test_default_64_bit_requires_device_support(device_type: spy.DeviceType) -> None:
    assert spy.ShaderCoverageOptions().counter_width == 64
    device = spy.Device(
        type=device_type, compiler_options={"coverage": spy.ShaderCoverageOptions()}
    )
    module = device.load_module_from_source(
        "coverage_width_probe", '[shader("compute")][numthreads(1,1,1)] void main() {}'
    )
    if device.has_feature(spy.Feature.atomic_int64):
        assert device.slang_session.link_program([module], [module.entry_point("main")])
    else:
        with pytest.raises(RuntimeError, match="64-bit shader coverage requires atomic_int64"):
            device.slang_session.link_program([module], [module.entry_point("main")])


def test_invalid_counter_width(device_type: spy.DeviceType) -> None:
    device = create_device(device_type, width=16)
    module = device.load_module_from_source(
        "coverage_invalid_width", '[shader("compute")][numthreads(1,1,1)] void main() {}'
    )
    with pytest.raises(RuntimeError, match="Coverage counter width must be 32 or 64"):
        device.slang_session.link_program([module], [module.entry_point("main")])


def test_64_bit_counters_cross_uint32_boundary(device_type: spy.DeviceType) -> None:
    probe = create_device(device_type, enabled=False)
    if not probe.has_feature(spy.Feature.atomic_int64):
        pytest.skip("This device lacks 64-bit buffer atomics")
    device = create_device(device_type, width=64)
    module = spy.Module.load_from_file(device, str(SOURCE))
    inputs = spy.Tensor.from_numpy(device, np.array([-3, -1, 0, 2], dtype=np.int32))
    output = spy.Tensor.from_numpy(device, np.zeros(4, dtype=np.int32))
    module.classify(inputs, _result=output)
    program = next(iter(module.pipeline_cache.values())).program
    manifest, first = read_coverage(program)
    assert manifest["buffer"]["element_stride"] == 8
    seed = np.full_like(first, 2**32 - 1)
    program.coverage_buffer.copy_from_numpy(seed)
    module.classify(inputs, _result=output)
    _, actual = read_coverage(program)
    np.testing.assert_array_equal(actual, seed + first)
    assert actual.max() > 2**32
    snapshot = device.shader_coverage.snapshot()
    assert snapshot.programs[0].counter_width == 64
    assert snapshot.programs[0].counters == [int(value) for value in actual]
