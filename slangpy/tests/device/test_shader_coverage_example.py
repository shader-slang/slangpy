# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the coverage progression promised by the image-processing example."""

import json
from pathlib import Path

import pytest
import slangpy as spy
from slangpy.testing import helpers
from examples.shader_coverage.image_processing import SOURCE, ImageProcessor, make_inputs
from examples.shader_coverage.report import save_capture, summarize
from examples.shader_coverage.shader_coverage import main


@pytest.mark.parametrize(
    "kinds",
    [("line", "branch", "function"), ("line",), ("branch",), ("function",), ("branch", "function")],
)
@pytest.mark.parametrize("boolean", [False, True])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_image_example_coverage(
    device_type: spy.DeviceType, tmp_path: Path, boolean: bool, kinds: tuple[str, ...]
) -> None:
    if device_type not in (spy.DeviceType.vulkan, spy.DeviceType.cuda):
        pytest.skip("Shader coverage supports Vulkan and CUDA")
    scenarios = make_inputs()
    with spy.Device(
        type=device_type,
        enable_hot_reload=False,
        compiler_options={
            "coverage": {
                "counter_width": 32,
                "boolean": boolean,
                "lines": "line" in kinds,
                "branches": "branch" in kinds,
                "functions": "function" in kinds,
            }
        },
    ) as device:
        processor = ImageProcessor(device)
        _, image, denoise = scenarios[0]
        processor.process(image, denoise)
        before = device.shader_coverage.snapshot()
        for _, image, denoise in scenarios[1:]:
            processor.process(image, denoise)
        after = device.shader_coverage.snapshot(reset=True)
        cleared = device.shader_coverage.snapshot()
        assert all(not any(program.counters) for program in cleared.programs)

    # Export also exercises the example's use of snapshots after closing the device.
    stats = []
    for label, snapshot, calls in (("before", before, 1), ("after", after, 3)):
        capture = save_capture(tmp_path, label, snapshot, SOURCE.read_text(encoding="utf-8"))
        current = summarize(capture, SOURCE)
        assert {entry["kind"] for entry in capture["manifest"]["entries"]} == set(kinds)
        if "function" in kinds:
            assert int(current["function_calls"]["denoiseToneMap"]) == (
                1 if boolean else calls * 240 * 320
            )
        stats.append(current)
    first, last = stats
    assert first["generation_id"] == last["generation_id"]
    for kind in ("line", "branch", "function"):
        if kind not in kinds:
            assert first[kind] == last[kind] == {"hit": 0, "total": 0}
        else:
            assert last[kind]["hit"] == last[kind]["total"] > 0
            if kind != "function":
                assert first[kind]["hit"] < first[kind]["total"]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_example_without_coverage(device_type: spy.DeviceType, tmp_path: Path, monkeypatch) -> None:
    if device_type not in (spy.DeviceType.vulkan, spy.DeviceType.cuda):
        pytest.skip("Example supports Vulkan and CUDA")
    monkeypatch.setattr(
        "sys.argv",
        [
            "shader_coverage.py",
            "--device",
            device_type.name,
            "--no-coverage",
            "--slang-source",
            str(tmp_path / "missing-slang"),
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["coverage"] == {}
    assert summary["coverage_kinds"] == []
    assert not (tmp_path / "ordinary_only.json").exists()
    assert not (tmp_path / "expanded_inputs.json").exists()
    assert not (tmp_path / "coverage").exists()
    page = (tmp_path / "index.html").read_text()
    assert "Coverage disabled" in page
    assert "Open coverage report" not in page
    for name in ("ordinary", "hdr_alpha", "filter_off"):
        assert (tmp_path / f"{name}-output.png").is_file()
