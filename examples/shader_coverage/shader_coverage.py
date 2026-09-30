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
        ImageProcessor,
        make_inputs,
        save_preview,
        validate_outputs,
    )
    from .report import render_capture, save_capture, summarize, write_report
else:
    from image_processing import SOURCE, ImageProcessor, make_inputs, save_preview, validate_outputs
    from report import render_capture, save_capture, summarize, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("vulkan", "cuda"), default="vulkan")
    parser.add_argument("--counter-width", type=int, choices=(32, 64), default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("shader-coverage-report"))
    parser.add_argument("--slang-source", type=Path, help="Slang checkout for LCOV/HTML rendering")
    parser.add_argument("--open", action="store_true", help="Open the image/report landing page")
    args = parser.parse_args()
    slang_source = args.slang_source.resolve() if args.slang_source else None
    if slang_source:
        for tool in (
            "tools/shader-coverage/slang-coverage-to-lcov.py",
            "tools/coverage-html/slang-coverage-html.py",
        ):
            if not (slang_source / tool).is_file():
                parser.error(f"Missing {slang_source / tool}")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source_text = SOURCE.read_text(encoding="utf-8")
    scenarios = make_inputs()
    outputs = {}
    device_type = spy.DeviceType[args.device]

    # Enable instrumentation before loading any shaders. The workload itself is unchanged.
    with spy.Device(
        type=device_type,
        enable_hot_reload=False,
        compiler_options={"coverage": spy.ShaderCoverageOptions(counter_width=args.counter_width)},
    ) as device:
        processor = ImageProcessor(device)
        name, image, denoise = scenarios[0]
        outputs[name] = processor.process(image, denoise)
        before = device.shader_coverage.snapshot()

        # Exercise the paths missed by the ordinary image, using the same compiled program.
        for name, image, denoise in scenarios[1:]:
            outputs[name] = processor.process(image, denoise)
        after = device.shader_coverage.snapshot(reset=True)

    # Snapshots own host data, so export works after the device is closed. The second
    # capture includes the first: compare these cumulative snapshots, never add them.
    if SOURCE.read_text(encoding="utf-8") != source_text:
        raise RuntimeError("Shader source changed while the example was running")
    summary = {"backend": args.device, "counter_width": args.counter_width, "coverage": {}}
    for label, snapshot in (("ordinary_only", before), ("expanded_inputs", after)):
        capture = save_capture(directory, label, snapshot, source_text)
        summary["coverage"][label] = summarize(capture, SOURCE)
        if slang_source:
            render_capture(directory, label, slang_source, SOURCE)

    # Numerical checks and image/report formatting are separate from coverage collection.
    summary["validation"] = validate_outputs(device_type, scenarios, outputs)
    for name, image, _ in scenarios:
        save_preview(directory / f"{name}-input.png", image, linear=True)
        save_preview(directory / f"{name}-output.png", outputs[name])
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(directory, summary, lcov_reports=slang_source is not None)
    for label, stats in summary["coverage"].items():
        print(f'{label}: lines {stats["line"]}, branch arms {stats["branch"]}')
    report = directory / "index.html"
    print(report.as_uri())
    if args.open:
        webbrowser.open(report.as_uri())


if __name__ == "__main__":
    main()
