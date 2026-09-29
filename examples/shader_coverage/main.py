# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Show image-processing shader coverage with a self-contained HTML report.

Run with --counter-width 32 on MoltenVK. The default is 64 bits.
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

SOURCE = Path(__file__).with_name("postprocess.slang").resolve()
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def make_inputs() -> list[tuple[str, np.ndarray, bool]]:
    """Generate reproducible edges, gradients, and noise without an image asset."""
    y, x = np.mgrid[:240, :320].astype(np.float32)
    x /= 319
    y /= 239
    rgb = np.stack((0.1 + 0.5 * x, 0.15 + 0.4 * y, 0.65 - 0.4 * x), axis=-1)
    rgb[(x - 0.65) ** 2 + (y - 0.35) ** 2 < 0.18**2] = [0.9, 0.25, 0.08]
    rgb[(x < 0.3) & (y > 0.35) & (y < 0.8)] = [0.05, 0.15, 0.75]
    noise = np.random.default_rng(739).normal(0, 0.025, rgb.shape).astype(np.float32)
    rgba = np.ones((240, 320, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(rgb + noise, 0.02, 0.95)
    stress = rgba.copy()
    stress[..., :3] *= 6.0
    stress[30:100, 20:100, :3] = 0
    stress[140:220, 220:300, 3] = 0
    return [("ordinary", rgba, True), ("hdr_alpha", stress, True), ("filter_off", stress, False)]


def reference(image: np.ndarray, denoise: bool) -> np.ndarray:
    """Independent NumPy implementation of the shader's 3x3 edge-aware filter."""
    color = image[..., :3].copy()
    height, width = color.shape[:2]
    if denoise:
        light = color @ LUMA
        total = np.zeros_like(color)
        weights = np.zeros((height, width), dtype=np.float32)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y = slice(max(0, -dy), min(height, height - dy))
                x = slice(max(0, -dx), min(width, width - dx))
                neighbor = color[
                    max(0, dy) : min(height, height + dy), max(0, dx) : min(width, width + dx)
                ]
                accepted = np.abs(neighbor @ LUMA - light[y, x]) <= 0.12
                weight = np.float32(1.0 / (1 + dx * dx + dy * dy))
                total[y, x] += neighbor * (accepted * weight)[..., None]
                weights[y, x] += accepted * weight
        color = total / weights[..., None]
    light = color @ LUMA
    color = color / np.maximum(light, 1.0)[..., None]
    color = np.sqrt(np.clip(color, 0, 1))
    color[light <= 1e-6] = 0
    output = image.copy()
    output[..., :3] = color
    output[image[..., 3] == 0] = image[image[..., 3] == 0]
    return output


def dispatch(
    device: spy.Device, module: spy.Module, image: np.ndarray, denoise: bool
) -> np.ndarray:
    height, width = image.shape[:2]
    texture = device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource,
        data=image,
    )
    output = device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    module.denoiseToneMap(spy.grid((height, width)), texture, 1.0, 0.12, denoise, _result=output)
    return output.to_numpy()


def save_preview(path: Path, rgba: np.ndarray, linear: bool = False) -> None:
    rgb = np.clip(rgba[..., :3], 0, 1)
    if linear:
        rgb = rgb ** (1.0 / 2.2)
    y, x = np.indices(rgb.shape[:2])
    checker = np.where((x // 12 + y // 12) % 2, 0.65, 0.85)[..., None]
    alpha = rgba[..., 3:4]
    rgb = rgb * alpha + checker * (1 - alpha)
    spy.Bitmap(np.rint(rgb * 255).astype(np.uint8)).write(path)


def summarize(program: spy.ShaderCoverageProgramSnapshot) -> dict[str, Any]:
    """Report just this example's source within one compiled program generation."""
    entries = [
        entry
        for entry in json.loads(program.manifest)["entries"]
        if entry.get("file") and Path(entry["file"]).resolve() == SOURCE
    ]
    lines: dict[int, bool] = {}
    branches: dict[tuple[int, int], dict[str, Any]] = {}
    functions: dict[str, int] = {}
    for entry in entries:
        hits = program.counters[entry["counter"]]
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
        "generation_id": program.generation_id,
        "line": {"hit": sum(lines.values()), "total": len(lines)},
        "branch": {
            "hit": sum(int(arm["hits"]) > 0 for arm in branches.values()),
            "total": len(branches),
        },
        "lines": lines,
        "arms": list(branches.values()),
        "function_calls": {name: str(hits) for name, hits in functions.items()},
    }


def write_report(
    directory: Path, summary: dict[str, Any], source: str, *, lcov_reports: bool = False
) -> None:
    sections = []
    for name, stats in summary["coverage"].items():
        if lcov_reports:
            sections.append(
                f'<section><h2>{html.escape(name.replace("_", " ").title())}</h2>'
                f'<p>Lines: {stats["line"]["hit"]}/{stats["line"]["total"]}. '
                f'Branch arms: {stats["branch"]["hit"]}/{stats["branch"]["total"]}.</p>'
                f'<p><a href="coverage/{name}/html/index.html">Open coverage report</a> '
                f'&middot; <a href="coverage/{name}/coverage.info">Download LCOV</a> '
                f'&middot; <a href="{name}.json">Raw capture</a></p></section>'
            )
            continue
        rows = []
        for line, text in enumerate(source.splitlines(), 1):
            hit = stats["lines"].get(line)
            color = "hit" if hit else "miss" if hit is False else ""
            arms = "; ".join(
                f'site {arm["site"]}, {arm["kind"]}: {arm["hits"]}'
                for arm in stats["arms"]
                if arm["line"] == line
            )
            rows.append(
                f'<tr class="{color}"><td>{line}</td><td><pre>{html.escape(text)}</pre></td>'
                f"<td>{html.escape(arms)}</td></tr>"
            )
        sections.append(
            f'<section><h2>{html.escape(name.replace("_", " ").title())}</h2>'
            f'<p>Lines: {stats["line"]["hit"]}/{stats["line"]["total"]}. '
            f'Branch arms: {stats["branch"]["hit"]}/{stats["branch"]["total"]}.</p>'
            "<details><summary>Source and branch hit counts</summary><table>"
            "<tr><th>Line</th><th>Source</th><th>Branch arms</th></tr>"
            + "".join(rows)
            + "</table></details></section>"
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
    (directory / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>SlangPy shader coverage</title><style>"
        "body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;"
        "background:#101820;color:#e6eef5}a{color:#8dccff}p{line-height:1.6}"
        ".pair{display:flex;gap:20px}"
        "figure{margin:0;flex:1;min-width:0}img{width:100%}figcaption{padding:8px 0}"
        "section{margin:32px 0}details{overflow:auto}summary{cursor:pointer}"
        "table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 8px;text-align:left}"
        "td pre{margin:0}.hit{background:#12382d}.miss{background:#562b2f}"
        "</style></head><body><h1>SlangPy shader coverage</h1>"
        "<p>A 3x3 edge-preserving denoiser, HDR normalization, and display gamma. "
        "The first capture uses only the ordinary image; the second accumulates all three inputs. "
        "Each image is 320 x 240 pixels. HDR previews are clipped; transparency uses a checkerboard.</p>"
        f'<p>Backend: {html.escape(summary["backend"])}. Counters: {summary["counter_width"]} bits. '
        "Branch counts belong to one compiled program.</p>"
        + "".join(sections)
        + figures
        + "<h2>Numerical validation</h2><pre>"
        + html.escape(json.dumps(summary["validation"], indent=2))
        + "</pre></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("vulkan", "cuda"), default="vulkan")
    parser.add_argument("--counter-width", type=int, choices=(32, 64), default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("shader-coverage-report"))
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source_text = SOURCE.read_text(encoding="utf-8")
    scenarios = make_inputs()
    outputs: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "backend": args.device,
        "counter_width": args.counter_width,
        "coverage": {},
        "validation": {},
    }
    device = spy.Device(
        type=spy.DeviceType[args.device],
        enable_hot_reload=False,
        compiler_options={"coverage": spy.ShaderCoverageOptions(counter_width=args.counter_width)},
    )
    try:
        module = spy.Module.load_from_file(device, str(SOURCE))
        for index, (name, image, denoise) in enumerate(scenarios):
            actual = dispatch(device, module, image, denoise)
            expected = reference(image, denoise)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            outputs[name] = actual
            summary["validation"][name] = {
                "numpy_max_error": float(np.max(np.abs(actual - expected)))
            }
            save_preview(directory / f"{name}-input.png", image, linear=True)
            save_preview(directory / f"{name}-output.png", actual)
            # Cumulative captures: do not sum these overlapping snapshots.
            snapshot = device.shader_coverage.snapshot(reset=index == len(scenarios) - 1)
            assert len(snapshot.programs) == 1, "Expected one compiled program for all inputs"
            program = snapshot.programs[0]
            if index in (0, len(scenarios) - 1):
                label = "ordinary_only" if index == 0 else "expanded_inputs"
                stats = summarize(program)
                assert (
                    int(stats["function_calls"]["denoiseToneMap"])
                    == (index + 1) * image.shape[0] * image.shape[1]
                )
                summary["coverage"][label] = stats
                # Example-specific dump; strings preserve counter precision in JSON readers.
                dump = {
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
                (directory / f"{label}.json").write_text(
                    json.dumps(dump, indent=2) + "\n", encoding="utf-8"
                )
        assert not any(device.shader_coverage.snapshot().programs[0].counters)
    finally:
        device.close()

    device = spy.Device(type=spy.DeviceType[args.device], enable_hot_reload=False)
    try:
        module = spy.Module.load_from_file(device, str(SOURCE))
        for name, image, denoise in scenarios:
            plain = dispatch(device, module, image, denoise)
            np.testing.assert_allclose(outputs[name], plain, rtol=2e-5, atol=2e-5)
            summary["validation"][name]["uninstrumented_max_error"] = float(
                np.max(np.abs(outputs[name] - plain))
            )
    finally:
        device.close()
    assert (
        SOURCE.read_text(encoding="utf-8") == source_text
    ), "Source changed while the example was running"
    before, after = summary["coverage"].values()
    assert before["generation_id"] == after["generation_id"]
    assert after["branch"]["hit"] > before["branch"]["hit"]
    assert after["branch"]["hit"] == after["branch"]["total"]
    assert after["line"]["hit"] == after["line"]["total"]
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(directory, summary, source_text)
    for label, stats in summary["coverage"].items():
        print(
            f'{label}: lines {stats["line"]["hit"]}/{stats["line"]["total"]}, '
            f'branch arms {stats["branch"]["hit"]}/{stats["branch"]["total"]}'
        )
    print(f"Report: {directory / 'index.html'}")


if __name__ == "__main__":
    main()
