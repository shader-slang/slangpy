# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Measure shader coverage around ordinary SlangPy image-processing calls."""

import argparse
import json
from pathlib import Path
import webbrowser

import slangpy as spy

if __package__:
    from .image_processing import (
        SOURCE,
        SOURCES,
        ImageProcessor,
        make_inputs,
        save_preview,
        validate_outputs,
    )
    from .report import render_capture, save_capture, summarize, write_report
else:
    from image_processing import (
        SOURCE,
        SOURCES,
        ImageProcessor,
        make_inputs,
        save_preview,
        validate_outputs,
    )
    from report import render_capture, save_capture, summarize, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("vulkan", "cuda"), default="vulkan")
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument(
        "--boolean",
        dest="boolean",
        action="store_true",
        help="Record hits instead of execution counts",
    )
    recording.add_argument(
        "--count",
        dest="boolean",
        action="store_false",
        help="Count executions (default)",
    )
    parser.add_argument(
        "--coverage",
        nargs="+",
        choices=("line", "branch", "function"),
        default=["line", "branch", "function"],
        help="Coverage kinds to instrument (default: all three)",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable instrumentation and collection; produce images and validation only",
    )
    parser.add_argument("--counter-width", type=int, choices=(32, 64), default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("shader-coverage-report"))
    parser.add_argument("--slang-source", type=Path, help="Slang checkout for LCOV/HTML rendering")
    parser.add_argument("--open", action="store_true", help="Open the image/report landing page")
    args = parser.parse_args()
    slang_source = args.slang_source.resolve() if args.slang_source else None
    if slang_source and not args.no_coverage:
        for tool in (
            "tools/shader-coverage/slang-coverage-to-lcov.py",
            "tools/coverage-html/slang-coverage-html.py",
        ):
            if not (slang_source / tool).is_file():
                parser.error(f"Missing {slang_source / tool}")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sources = {str(path): path.read_text(encoding="utf-8") for path in SOURCES}
    scenarios = make_inputs()
    outputs = {}
    device_type = spy.DeviceType[args.device]

    # None disables instrumentation. The same workload runs in either case.
    coverage = None
    if not args.no_coverage:
        coverage = spy.ShaderCoverageOptions(
            lines="line" in args.coverage,
            branches="branch" in args.coverage,
            functions="function" in args.coverage,
            counter_width=args.counter_width,
            boolean=args.boolean,
        )
    with spy.Device(
        type=device_type,
        enable_hot_reload=False,
        compiler_options={"coverage": coverage},
    ) as device:
        collector = device.shader_coverage if coverage is not None else None
        processor = ImageProcessor(device)
        name, image, denoise, mapper_id = scenarios[0]
        outputs[name] = processor.process(image, denoise, mapper_id)
        before = collector.snapshot() if collector is not None else None
        checkpoints = [(name, before)]

        # Exercise the paths missed by the ordinary image, using the same compiled program.
        for index, (name, image, denoise, mapper_id) in enumerate(scenarios[1:], start=1):
            outputs[name] = processor.process(image, denoise, mapper_id)
            snapshot = (
                collector.snapshot(reset=index == len(scenarios) - 1)
                if collector is not None
                else None
            )
            checkpoints.append((name, snapshot))

    # Snapshots own host data, so export works after the device is closed.
    # Captures are cumulative: compare their totals, never add the snapshots.
    for path, text in sources.items():
        if Path(path).read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Shader source changed while the example was running: {path}")
    summary = {
        "backend": args.device,
        "counter_width": args.counter_width,
        "boolean": args.boolean,
        "coverage_kinds": args.coverage if coverage is not None else [],
        "coverage": {},
        "progress": [],
        "scenarios": [name for name, _, _, _ in scenarios],
    }
    for index, (name, snapshot) in enumerate(checkpoints):
        if snapshot is None:
            continue
        label = (
            "ordinary_only"
            if index == 0
            else "expanded_inputs" if index == len(checkpoints) - 1 else f"after_{name}"
        )
        capture = save_capture(directory, label, snapshot, sources)
        stats = summarize(capture)
        summary["progress"].append(
            {
                "name": name,
                "capture": label,
                **{kind: stats[kind] for kind in ("line", "branch", "function")},
            }
        )
        if index in (0, len(checkpoints) - 1):
            summary["coverage"][label] = stats
            if slang_source:
                render_capture(directory, label, slang_source, SOURCE)

    # Numerical checks and image/report formatting are separate from coverage collection.
    summary["validation"] = validate_outputs(device_type, scenarios, outputs)
    for name, image, _, _ in scenarios:
        save_preview(directory / f"{name}-input.png", image, linear=True)
        save_preview(directory / f"{name}-output.png", outputs[name])
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(directory, summary, lcov_reports=slang_source is not None)
    for label, stats in summary["coverage"].items():
        print(f"{label}: " + ", ".join(f"{kind} {stats[kind]}" for kind in args.coverage))
    report = directory / "index.html"
    print(report.as_uri())
    if args.open:
        webbrowser.open(report.as_uri())


if __name__ == "__main__":
    main()
