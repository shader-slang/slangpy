# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Example-specific snapshot export and reporting; importing this needs no GPU runtime."""

from __future__ import annotations

import html
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import slangpy as spy


def save_capture(
    directory: Path, name: str, snapshot: spy.ShaderCoverageSnapshot, source_text: str
) -> dict[str, Any]:
    """Save the example's single program without losing 64-bit counter precision."""
    if len(snapshot.programs) != 1:
        raise ValueError("This example expects one compiled program per capture")
    program = snapshot.programs[0]
    capture = {
        "collection_id": snapshot.collection_id,
        "capture_id": snapshot.capture_id,
        "interval_id": snapshot.interval_id,
        "reset_after": snapshot.reset_after,
        "generation_id": program.generation_id,
        "counter_width": program.counter_width,
        "manifest": json.loads(program.manifest),
        "counters": [str(value) for value in program.counters],
        "source": source_text,
    }
    (directory / f"{name}.json").write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")
    return capture


def summarize(capture: dict[str, Any], source: Path) -> dict[str, Any]:
    """Report just this example's source within one compiled program generation."""
    entries = [
        entry
        for entry in capture["manifest"]["entries"]
        if entry.get("file") and Path(entry["file"]).resolve() == source
    ]
    lines: dict[int, bool] = {}
    branches: dict[tuple[int, int], dict[str, Any]] = {}
    functions: dict[str, int] = {}
    for entry in entries:
        hits = int(capture["counters"][entry["counter"]])
        if entry["kind"] == "line":
            line = entry["line"]
            lines[line] = lines.get(line, False) or hits > 0
        elif entry["kind"] == "branch":
            # IDs are local to this program. Coordinates alone can merge distinct macro expansions.
            key = (entry["branch_site"], entry["branch_arm"])
            if key in branches:
                raise RuntimeError("Unexpected duplicate branch arm in example manifest")
            branches[key] = {
                "site": key[0],
                "arm": key[1],
                "kind": entry["branch_arm_kind"],
                "line": entry["line"],
                "hits": str(hits),
            }
        elif entry["kind"] == "function":
            functions[entry["function"]] = hits
    return {
        "generation_id": capture["generation_id"],
        "line": {"hit": sum(lines.values()), "total": len(lines)},
        "branch": {
            "hit": sum(int(arm["hits"]) > 0 for arm in branches.values()),
            "total": len(branches),
        },
        "function": {"hit": sum(hits > 0 for hits in functions.values()), "total": len(functions)},
        "lines": lines,
        "arms": list(branches.values()),
        "function_calls": {name: str(hits) for name, hits in functions.items()},
    }


def write_report(directory: Path, summary: dict[str, Any], *, lcov_reports: bool = False) -> None:
    sections = []
    for name, stats in summary["coverage"].items():
        links = f'<a href="{name}.json">Raw capture</a>'
        if lcov_reports:
            links += (
                f' &middot; <a href="coverage/{name}/html/index.html">Open coverage report</a>'
                f' &middot; <a href="coverage/{name}/coverage.info">Download LCOV</a>'
            )
        totals = " ".join(
            (
                f'{kind.title()}: {stats[kind]["hit"]}/{stats[kind]["total"]}.'
                if kind in summary["coverage_kinds"]
                else f"{kind.title()}: disabled."
            )
            for kind in ("line", "branch", "function")
        )
        sections.append(
            f'<section><h2>{html.escape(name.replace("_", " ").title())}</h2>'
            f"<p>{totals}</p>"
            f"<p>{links}</p></section>"
        )
    figures = "".join(
        f'<section><h2>{title}</h2><div class="pair"><figure>'
        f'<img src="{name}-input.png" alt="{title}: input"><figcaption>Input</figcaption></figure>'
        f'<figure><img src="{name}-output.png" alt="{title}: GPU output">'
        "<figcaption>GPU output</figcaption></figure></div></section>"
        for name, title in (
            ("ordinary", "Noisy image"),
            ("hdr_alpha", "HDR and transparency"),
            ("filter_off", "Denoising disabled"),
        )
    )
    if summary["coverage_kinds"]:
        configuration = (
            f'Counters: {summary["counter_width"]} bits. '
            f'Recording: {"boolean (hit/miss)" if summary["boolean"] else "execution counts"}. '
            f'Coverage: {html.escape(", ".join(summary["coverage_kinds"]))}. '
            "Branch records belong to one compiled program."
        )
    else:
        configuration = "Coverage disabled. No instrumentation or coverage captures."
    (directory / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>SlangPy shader coverage</title><style>"
        "body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;"
        "background:#101820;color:#e6eef5}a{color:#8dccff}p{line-height:1.6}"
        ".pair{display:flex;gap:20px}"
        "figure{margin:0;flex:1;min-width:0}img{width:100%}figcaption{padding:8px 0}"
        "section{margin:32px 0}"
        "</style></head><body><h1>SlangPy shader coverage</h1>"
        "<p>A 3x3 edge-preserving denoiser, HDR normalization, and display gamma. "
        "The workload processes an ordinary image, HDR/transparency, and a denoising bypass. "
        "When enabled, coverage compares the first input with all three accumulated inputs. "
        "Each image is 320 x 240 pixels. HDR previews are clipped; transparency uses a checkerboard.</p>"
        f'<p>Backend: {html.escape(summary["backend"])}. {configuration}</p>'
        + "".join(sections)
        + figures
        + "<h2>Numerical validation</h2><pre>"
        + html.escape(json.dumps(summary["validation"], indent=2))
        + "</pre></body></html>",
        encoding="utf-8",
    )


def render_capture(directory: Path, name: str, slang_source: Path, source: Path) -> None:
    """Convert one cumulative capture without combining overlapping snapshots."""
    capture = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    if capture["source"] != source.read_text(encoding="utf-8"):
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
            str(source.parent),
            "--filter-include",
            "*postprocess.slang",
            "--title",
            f"SlangPy shader coverage: {name.replace('_', ' ')}",
        ],
        check=True,
    )
