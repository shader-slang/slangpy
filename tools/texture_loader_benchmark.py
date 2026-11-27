# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Benchmark TextureLoader file-to-GPU throughput.

Build a dataset once from a directory of images, then run the benchmark against it and compare runs:

    python tools/texture_loader_benchmark.py make-dataset --source <images dir> --out <dataset dir>
    python tools/texture_loader_benchmark.py run --dataset <dataset dir> --label baseline --json base.json
    python tools/texture_loader_benchmark.py compare base.json new.json

The dataset contains JPG and PNG files copied from the source, and EXR files copied from the source
or converted from its HDR and largest 8-bit PNG images (PIZ and DWAB compression). Each run loads
every set with one load_textures() call, waits for the GPU and reports the median over repeats.
Repeats after the first read warm OS file caches; the first repeat is reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import slangpy as spy

SETS = ["jpg", "png", "exr", "mixed"]

# Number of largest 8-bit PNG images converted to EXR.
PNG_TO_EXR_COUNT = 8
# DWAB compression level used for converted EXR images.
DWAB_QUALITY = 45


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_unique(source: Path, extensions: tuple[str, ...]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        digest = file_hash(path)
        if digest in seen:
            continue
        seen.add(digest)
        result.append(path)
    return result


def write_exr(bitmap: spy.Bitmap, path: Path, quality: int) -> None:
    pixel_format = (
        spy.Bitmap.PixelFormat.rgba
        if bitmap.pixel_format == spy.Bitmap.PixelFormat.rgba
        else spy.Bitmap.PixelFormat.rgb
    )
    half = bitmap.convert(pixel_format, spy.Bitmap.ComponentType.float16, False)
    half.write(path, spy.Bitmap.FileFormat.exr, quality)


def make_dataset(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        if not args.force:
            sys.exit(f"Output directory {out} exists; pass --force to replace it.")
        shutil.rmtree(out)
    for name in ("jpg", "png", "exr"):
        (out / name).mkdir(parents=True)

    def copy(path: Path, set_name: str) -> None:
        shutil.copy2(path, out / set_name / f"{file_hash(path)[:8]}_{path.name}")

    jpgs = collect_unique(source, (".jpg", ".jpeg"))
    pngs = collect_unique(source, (".png",))
    exrs = collect_unique(source, (".exr",))
    hdrs = collect_unique(source, (".hdr",))
    for path in jpgs:
        copy(path, "jpg")
    for path in pngs:
        copy(path, "png")
    for path in exrs:
        copy(path, "exr")

    for path in hdrs:
        bitmap = spy.Bitmap(path)
        stem = f"{file_hash(path)[:8]}_{path.stem}"
        write_exr(bitmap, out / "exr" / f"{stem}_piz.exr", -1)
        write_exr(bitmap, out / "exr" / f"{stem}_dwab.exr", DWAB_QUALITY)

    # Convert the largest 8-bit color PNG images, which approximates high-resolution EXR textures.
    candidates: list[tuple[int, Path]] = []
    for path in pngs:
        bitmap = spy.Bitmap(path)
        if bitmap.component_type != spy.Bitmap.ComponentType.uint8:
            continue
        if bitmap.pixel_format not in (spy.Bitmap.PixelFormat.rgb, spy.Bitmap.PixelFormat.rgba):
            continue
        candidates.append((bitmap.width * bitmap.height, path))
    candidates.sort(key=lambda c: (-c[0], str(c[1])))
    for _, path in candidates[:PNG_TO_EXR_COUNT]:
        bitmap = spy.Bitmap(path)
        stem = f"{file_hash(path)[:8]}_{path.stem}"
        write_exr(bitmap, out / "exr" / f"{stem}_piz.exr", -1)
        write_exr(bitmap, out / "exr" / f"{stem}_dwab.exr", DWAB_QUALITY)

    manifest: dict[str, Any] = {"source": str(source), "sets": {}}
    for name in ("jpg", "png", "exr"):
        files = sorted((out / name).iterdir())
        entries = []
        for path in files:
            info = spy.Bitmap.read_info(path)
            entries.append(
                {
                    "file": f"{name}/{path.name}",
                    "bytes": path.stat().st_size,
                    "width": info.width,
                    "height": info.height,
                    "pixel_format": str(info.pixel_format),
                    "component_type": str(info.component_type),
                }
            )
        manifest["sets"][name] = entries
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for name, entries in manifest["sets"].items():
        size = sum(e["bytes"] for e in entries)
        pixels = sum(e["width"] * e["height"] for e in entries)
        print(f"{name}: {len(entries)} files, {size / 2**20:.1f} MiB, {pixels / 1e6:.1f} MPix")
    print(f"Dataset written to {out}")


def git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def time_load(load: Callable[[], Any], device: spy.Device) -> float:
    start = time.perf_counter()
    textures = load()
    device.wait()
    elapsed = time.perf_counter() - start
    del textures
    device.wait()
    return elapsed


def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    sets: dict[str, list[dict[str, Any]]] = dict(manifest["sets"])
    sets["mixed"] = [e for name in ("jpg", "png", "exr") for e in manifest["sets"][name]]

    device = spy.Device(type=spy.DeviceType[args.device_type])
    loader = spy.TextureLoader(device)
    options = spy.TextureLoader.Options()
    options.srgb_mode = spy.SRGBMode.srgb
    options.y_handling = spy.YHandling.expand_to_rgba
    options.generate_mips = not args.no_mips
    options.max_mip_count = args.max_mip_count

    # Warm up shader compilation for mip generation before timing.
    first = dataset / sets["jpg"][0]["file"] if sets["jpg"] else None
    if first is not None:
        time_load(lambda: loader.load_textures([first], options), device)

    results: dict[str, Any] = {}
    for name in args.sets:
        entries = sets[name]
        paths = [dataset / e["file"] for e in entries]
        encoded_bytes = sum(e["bytes"] for e in entries)
        pixels = sum(e["width"] * e["height"] for e in entries)
        times = [
            time_load(lambda: loader.load_textures(paths, options), device)
            for _ in range(args.repeat)
        ]
        warm = times[1:] if len(times) > 1 else times
        median = statistics.median(warm)
        results[name] = {
            "files": len(entries),
            "encoded_bytes": encoded_bytes,
            "pixels": pixels,
            "first_s": times[0],
            "median_s": median,
            "min_s": min(warm),
            "times_s": times,
        }
        print(
            f"{name:6s} {len(entries):4d} files  first {times[0]:7.3f} s  "
            f"median {median:7.3f} s  min {min(warm):7.3f} s  "
            f"{encoded_bytes / 2**20 / median:8.1f} MiB/s encoded  {pixels / 1e6 / median:8.1f} MPix/s"
        )

    report = {
        "label": args.label,
        "slangpy_revision": git_revision(Path(__file__).resolve().parent.parent),
        "device_type": args.device_type,
        "adapter": device.info.adapter_name,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "generate_mips": not args.no_mips,
        "max_mip_count": args.max_mip_count,
        "repeat": args.repeat,
        "dataset": str(dataset),
        "results": results,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"Results written to {args.json}")


def compare(args: argparse.Namespace) -> None:
    base = json.loads(Path(args.base).read_text())
    new = json.loads(Path(args.new).read_text())
    print(f"base: {base['label']} ({base['slangpy_revision']}, {base['device_type']})")
    print(f"new:  {new['label']} ({new['slangpy_revision']}, {new['device_type']})")
    print(f"{'set':6s} {'base median':>12s} {'new median':>12s} {'speedup':>8s}")
    for name, b in base["results"].items():
        n = new["results"].get(name)
        if n is None:
            continue
        speedup = b["median_s"] / n["median_s"]
        print(f"{name:6s} {b['median_s']:11.3f}s {n['median_s']:11.3f}s {speedup:7.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-dataset", help="Build a benchmark dataset from a directory of images.")
    p.add_argument("--source", required=True, help="Directory searched recursively for images.")
    p.add_argument("--out", required=True, help="Output dataset directory.")
    p.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    p.set_defaults(func=make_dataset)

    p = sub.add_parser("run", help="Run the benchmark.")
    p.add_argument("--dataset", required=True, help="Dataset directory created by make-dataset.")
    p.add_argument("--device-type", default="d3d12" if sys.platform == "win32" else "vulkan")
    p.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    p.add_argument(
        "--repeat", type=int, default=5, help="Loads per set; the first is reported separately."
    )
    p.add_argument("--no-mips", action="store_true", help="Do not generate mip levels.")
    p.add_argument(
        "--max-mip-count", type=int, default=0, help="Mip count cap; zero is unrestricted."
    )
    p.add_argument("--label", default="", help="Label stored in the JSON report.")
    p.add_argument("--json", help="Write results to this JSON file.")
    p.set_defaults(func=run)

    p = sub.add_parser("compare", help="Compare two JSON reports.")
    p.add_argument("base")
    p.add_argument("new")
    p.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
