# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the coverage progression promised by the image-processing example."""

import json
from pathlib import Path

import pytest
import slangpy as spy
from slangpy.testing import helpers
from examples.shader_coverage.image_processing import (
    SOURCES,
    ImageProcessor,
    make_inputs,
    validate_outputs,
)
from examples.shader_coverage.report import save_capture, summarize, render_capture
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
    outputs = {}
    checkpoints = []
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
        name, image, denoise, mapper_id = scenarios[0]
        outputs[name] = processor.process(image, denoise, mapper_id)
        before = device.shader_coverage.snapshot()
        checkpoints.append(before)
        for name, image, denoise, mapper_id in scenarios[1:]:
            outputs[name] = processor.process(image, denoise, mapper_id)
            checkpoints.append(device.shader_coverage.snapshot())
        after = device.shader_coverage.snapshot(reset=True)
        cleared = device.shader_coverage.snapshot()
        assert all(not any(program.counters) for program in cleared.programs)

    validate_outputs(device_type, scenarios, outputs)

    # Export also exercises the example's use of snapshots after closing the device.
    stats = []
    for label, snapshot in (("before", before), ("after", after)):
        capture = save_capture(
            tmp_path, label, snapshot, {str(p): p.read_text(encoding="utf-8") for p in SOURCES}
        )
        current = summarize(capture)
        assert {entry["kind"] for entry in capture["manifest"]["entries"]} == set(kinds)
        if boolean:
            assert set(snapshot.programs[0].counters) <= {0, 1}
        else:
            assert max(snapshot.programs[0].counters) > 1
        assert set(current["files"]) == {p.name for p in SOURCES}
        stats.append(current)
    first, last = stats
    assert first["generation_id"] == last["generation_id"]
    for kind in ("line", "branch", "function"):
        if kind not in kinds:
            assert first[kind] == last[kind] == {"hit": 0, "total": 0}
        else:
            assert last[kind]["hit"] == last[kind]["total"] > 0
            assert first[kind]["hit"] < first[kind]["total"]
    if "branch" in kinds:
        assert last["branch"]["total"] >= 50
        assert first["branch"]["hit"] / first["branch"]["total"] < 0.75
        branch_counts = []
        for checkpoint in checkpoints:
            program = checkpoint.programs[0]
            manifest = json.loads(program.manifest)
            branch_counts.append(
                sum(
                    program.counters[e["counter"]] > 0
                    for e in manifest["entries"]
                    if e["kind"] == "branch"
                )
            )
        assert branch_counts == sorted(branch_counts)
        assert sum(b > a for a, b in zip(branch_counts, branch_counts[1:])) >= 4

    if "function" in kinds:
        first_maps = first["files"]["coverage_tonemap.slang"]["functions"]
        last_maps = last["files"]["coverage_tonemap.slang"]["functions"]
        assert sorted(int(f["hits"]) > 0 for f in first_maps if f["name"] == "map") == [
            False,
            False,
            True,
        ]
        assert sorted(int(f["hits"]) > 0 for f in last_maps if f["name"] == "map") == [
            True,
            True,
            True,
        ]
        assert any(
            f["name"] == "exposureLoss" and int(f["hits"]) > 0
            for f in last["files"]["coverage_exposure.slang"]["functions"]
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_example_without_coverage(
    device_type: spy.DeviceType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    for name, _, _, _ in make_inputs():
        assert (tmp_path / f"{name}-output.png").is_file()


def test_report_detects_changed_imported_source(tmp_path: Path) -> None:
    entry = tmp_path / "postprocess.slang"
    imported = tmp_path / "coverage_exposure.slang"
    entry.write_text("import coverage_exposure;", encoding="utf-8")
    imported.write_text("// original", encoding="utf-8")
    capture = {"sources": {str(entry): entry.read_text(), str(imported): imported.read_text()}}
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    imported.write_text("// changed", encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="Shader source changed since capture.*coverage_exposure"
    ):
        render_capture(tmp_path, "capture", tmp_path / "unused-tools", entry)


def test_report_tools_use_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report tools must preserve Unicode paths even with legacy Python defaults."""
    monkeypatch.setenv("PYTHONUTF8", "0")
    source = tmp_path / "\u7528\u6237.slang"
    source.write_text("// source", encoding="utf-8")
    capture = {
        "sources": {str(source): "// source"},
        "manifest": {"source": str(source)},
        "counters": ["1"],
    }
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    # Stand-ins use the same locale-dependent I/O as older Slang reporting tools.
    tools = tmp_path / "tools"
    converter = tools / "shader-coverage/slang-coverage-to-lcov.py"
    renderer = tools / "coverage-html/slang-coverage-html.py"
    converter.parent.mkdir(parents=True)
    renderer.parent.mkdir(parents=True)
    converter.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "assert sys.flags.utf8_mode, 'report tool must enable UTF-8 mode'\n"
        "args = sys.argv\n"
        "source = json.loads(Path(args[args.index('--manifest') + 1]).read_text())['source']\n"
        "Path(args[args.index('--output') + 1]).write_text('SF:' + source + '\\n')\n",
        encoding="utf-8",
    )
    renderer.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "assert sys.flags.utf8_mode, 'report tool must enable UTF-8 mode'\n"
        "output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "output.mkdir()\n"
        "(output / 'index.html').write_text(Path(sys.argv[1]).read_text())\n",
        encoding="utf-8",
    )
    render_capture(tmp_path, "capture", tmp_path, source)
    output = tmp_path / "coverage/capture"
    assert str(source) in (output / "coverage.info").read_text(encoding="utf-8")
    assert str(source) in (output / "html/index.html").read_text(encoding="utf-8")
