# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the image example and render its captures with Slang's LCOV tools."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import webbrowser

from .main import SOURCE, write_report


def render_capture(directory: Path, name: str, slang_source: Path) -> None:
    """Convert one cumulative capture without combining overlapping snapshots."""
    capture = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    if capture["source"] != SOURCE.read_text(encoding="utf-8"):
        raise RuntimeError("Shader source changed since capture; rerun the example")
    output = directory / "coverage" / name
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "coverage-manifest.json"
    counters = output / "counters.txt"
    lcov = output / "coverage.info"
    manifest.write_text(json.dumps(capture["manifest"], indent=2) + "\n", encoding="utf-8")
    counters.write_text(
        "\n".join(str(int(value)) for value in capture["counters"]) + "\n", encoding="utf-8"
    )
    subprocess.run(
        [
            sys.executable,
            str(slang_source / "tools/shader-coverage/slang-coverage-to-lcov.py"),
            "--manifest",
            str(manifest),
            "--counters-text",
            str(counters),
            "--output",
            str(lcov),
            "--test-name",
            name,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(slang_source / "tools/coverage-html/slang-coverage-html.py"),
            str(lcov),
            "--output-dir",
            str(output / "html"),
            "--source-root",
            str(SOURCE.parent),
            "--filter-include",
            "*postprocess.slang",
            "--title",
            f"SlangPy shader coverage: {name.replace('_', ' ')}",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("vulkan", "cuda"), default="vulkan")
    parser.add_argument("--counter-width", type=int, choices=(32, 64), default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("shader-coverage-report"))
    parser.add_argument(
        "--slang-source", type=Path, required=True, help="Slang checkout containing tools/"
    )
    parser.add_argument("--open", action="store_true", help="Open the landing page in a browser")
    args = parser.parse_args()
    slang_source = args.slang_source.resolve()
    for relative in (
        "tools/shader-coverage/slang-coverage-to-lcov.py",
        "tools/coverage-html/slang-coverage-html.py",
    ):
        if not (slang_source / relative).is_file():
            parser.error(
                f"Missing {slang_source / relative}; use a Slang checkout with coverage tools"
            )
    directory = args.output_dir.resolve()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.shader_coverage.main",
            "--device",
            args.device,
            "--counter-width",
            str(args.counter_width),
            "--output-dir",
            str(directory),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )
    for name in ("ordinary_only", "expanded_inputs"):
        render_capture(directory, name, slang_source)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    write_report(directory, summary, SOURCE.read_text(encoding="utf-8"), lcov_reports=True)
    report = directory / "index.html"
    print(f"Report with images and LCOV coverage links: {report}")
    print(report.as_uri())
    if args.open:
        webbrowser.open(report.as_uri())


if __name__ == "__main__":
    main()
