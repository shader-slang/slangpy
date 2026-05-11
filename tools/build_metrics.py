#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Build metrics analysis tool for SlangPy.

Analyzes build performance using multiple data sources:
  1. Ninja log (.ninja_log) - always available, shows wall-clock time per TU
  2. MSVC /Bt+ output - frontend/backend time per TU (requires SGL_ENABLE_BUILD_METRICS)
  3. Clang -ftime-trace JSON - per-TU traces with header/template detail (requires SGL_ENABLE_BUILD_METRICS)
  4. GCC -ftime-report output - per-pass timing summary (requires SGL_ENABLE_BUILD_METRICS)

Usage:
  # Analyze ninja log (always works, no special flags needed):
  python tools/build_metrics.py --build-dir build/windows-msvc

  # Analyze MSVC timing from captured build output:
  cmake --build --preset windows-msvc-debug 2>&1 | tee build_output.txt
  python tools/build_metrics.py --build-dir build/windows-msvc --log-file build_output.txt

  # Analyze Clang -ftime-trace JSON files:
  python tools/build_metrics.py --build-dir build/windows-clang

  # Generate Chrome trace from ninja log for parallelism visualization:
  python tools/build_metrics.py --build-dir build/windows-msvc --trace ninja_trace.json

Setup:
  cmake --preset <PLATFORM> -DSGL_ENABLE_BUILD_METRICS=ON
  # Or via ci.py:
  python tools/ci.py configure --flags build-metrics
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure stdout can handle unicode (box-drawing chars) on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


def parse_ninja_log(build_dir: Path, top_n: int) -> bool:
    """
    Parse .ninja_log and print the slowest translation units.

    :param build_dir: Path to the build directory containing .ninja_log.
    :param top_n: Number of top entries to display.
    :return: True if entries were found and displayed, False otherwise.
    """
    ninja_log = build_dir / ".ninja_log"
    if not ninja_log.exists():
        print(f"No .ninja_log found at {ninja_log}")
        return False

    entries: list[tuple[float, str]] = []
    with open(ninja_log, "r") as f:
        for line in f:
            # Skip header line
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            try:
                start_ms = int(parts[0])
                end_ms = int(parts[1])
                # parts[2] is mtime (restat), parts[3] is the output file
                output = parts[3] if len(parts) > 3 else parts[2]
                duration_s = (end_ms - start_ms) / 1000.0
                entries.append((duration_s, output))
            except (ValueError, IndexError):
                continue

    if not entries:
        print("No entries found in .ninja_log")
        return False

    # Filter to only compilation targets (.obj/.o) for the main report
    compile_entries = [(d, o) for d, o in entries if o.endswith((".obj", ".o"))]
    link_entries = [
        (d, o) for d, o in entries if o.endswith((".lib", ".dll", ".so", ".dylib", ".a", ".exe"))
    ]
    other_entries = [
        (d, o) for d, o in entries if (d, o) not in compile_entries and (d, o) not in link_entries
    ]

    # Sort compile entries by duration descending
    compile_entries.sort(reverse=True)
    link_entries.sort(reverse=True)

    total_time = sum(d for d, _ in entries)
    compile_time = sum(d for d, _ in compile_entries)
    link_time = sum(d for d, _ in link_entries)

    display_entries = compile_entries
    count = len(compile_entries)

    print(f"\n{'='*80}")
    print(f"NINJA LOG ANALYSIS - Top {min(top_n, count)} slowest compilation targets")
    print(f"{'='*80}")
    print(
        f"Compile time (sum of all TUs): {compile_time:.1f}s  |  Link time: {link_time:.1f}s  |  Total: {total_time:.1f}s"
    )
    print(
        f"Compilation targets: {count}  |  Link targets: {len(link_entries)}  |  Other: {len(other_entries)}"
    )
    print(f"{''*80}")
    print(f"{'Time (s)':>10}  {'%':>5}  {'Target'}")
    print(f"{''*80}")

    for duration, output in display_entries[:top_n]:
        pct = (duration / compile_time * 100) if compile_time > 0 else 0
        # Shorten path for display
        display = output
        if len(display) > 60:
            display = "..." + display[-57:]
        print(f"{duration:>10.2f}  {pct:>5.1f}  {display}")

    print(f"{''*80}")
    top_total = sum(d for d, _ in display_entries[:top_n])
    top_pct = (top_total / compile_time * 100) if compile_time > 0 else 0.0
    print(f"{'Top entries:':>10}  {top_total:.1f}s ({top_pct:.1f}% of compile time)")

    # Show link targets if any are significant
    if link_entries:
        print(f"\n{''*80}")
        print(f"Link targets:")
        for duration, output in link_entries[:5]:
            display = output
            if len(display) > 60:
                display = "..." + display[-57:]
            print(f"{duration:>10.2f}s  {display}")

    return True


def parse_msvc_bt(log_file: Path, top_n: int) -> bool:
    """
    Parse MSVC /Bt+ output from a captured build log.

    :param log_file: Path to the captured build output file.
    :param top_n: Number of top entries to display.
    :return: True if timing data was found and displayed, False otherwise.
    """
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return False

    # /Bt+ outputs lines like:
    #   time(C:\...\c1xx.dll)=3.568s < ... > BB [C:\path\to\file.cpp]
    #   time(C:\...\c2.dll)=0.289s < ... > BB [C:\path\to\file.cpp]
    # c1xx.dll = frontend, c2.dll = backend. Source file is in brackets at end.

    # Pattern for /Bt+ style output: time(<dll>)=<seconds>s ... [<source_file>]
    bt_pattern = re.compile(r"time\(([^)]+)\)\s*=\s*([\d.]+)s.*?\[([^\]]+)\]")

    file_times: dict[str, dict[str, float]] = {}

    with open(log_file, "r", errors="replace") as f:
        for line in f:
            m = bt_pattern.search(line)
            if m:
                dll_path = m.group(1)
                time_s = float(m.group(2))
                source_file = m.group(3).strip()

                if source_file not in file_times:
                    file_times[source_file] = {"total": 0.0, "frontend": 0.0, "backend": 0.0}

                file_times[source_file]["total"] += time_s

                # Determine if this is frontend (c1xx) or backend (c2)
                dll_name = os.path.basename(dll_path).lower()
                if "c1xx" in dll_name or "c1" in dll_name:
                    file_times[source_file]["frontend"] += time_s
                elif "c2" in dll_name:
                    file_times[source_file]["backend"] += time_s

    if not file_times:
        print("No MSVC /Bt+ timing data found in log file.")
        print("Make sure the build was configured with -DSGL_ENABLE_BUILD_METRICS=ON")
        print("and the build output (stderr) was captured to the log file.")
        return False

    # Sort by total time descending
    sorted_files = sorted(file_times.items(), key=lambda x: x[1]["total"], reverse=True)

    print(f"\n{'='*80}")
    print(f"MSVC /Bt+ ANALYSIS - Top {min(top_n, len(sorted_files))} slowest TUs")
    print(f"{'='*80}")
    print(f"{'Total':>8}  {'Front':>8}  {'Back':>8}  {'File'}")
    print(f"{''*80}")

    for filename, times in sorted_files[:top_n]:
        display = os.path.basename(filename) if len(filename) > 50 else filename
        print(
            f"{times['total']:>7.2f}s  {times['frontend']:>7.2f}s  {times['backend']:>7.2f}s  {display}"
        )

    print(f"{''*80}")
    total = sum(t["total"] for t in file_times.values())
    print(f"Total time across all TUs: {total:.1f}s")
    return True


def parse_ftime_trace(build_dir: Path, top_n: int) -> bool:
    """
    Parse Clang -ftime-trace JSON files and aggregate results.

    :param build_dir: Path to the build directory containing trace files.
    :param top_n: Number of top entries to display per category.
    :return: True if trace data was found and displayed, False otherwise.
    """
    # Find all .json trace files in the build tree
    trace_files: list[Path] = []
    for root, _dirs, files in os.walk(build_dir):
        for f in files:
            if f.endswith(".json"):
                filepath = Path(root) / f
                # Quick check: is this a trace file? (has traceEvents key)
                try:
                    with open(filepath, "r") as fh:
                        first_bytes = fh.read(100)
                        if "traceEvents" in first_bytes:
                            trace_files.append(filepath)
                except (OSError, UnicodeDecodeError):
                    continue

    if not trace_files:
        print("No Clang -ftime-trace JSON files found in build directory.")
        print("Make sure the build was configured with -DSGL_ENABLE_BUILD_METRICS=ON")
        print("and compiled with Clang.")
        return False

    print(f"\nFound {len(trace_files)} trace files. Analyzing...")

    # Aggregate: header parse times and template instantiation times
    header_times: dict[str, float] = {}
    template_times: dict[str, float] = {}
    source_times: dict[str, float] = {}

    for trace_file in trace_files:
        try:
            with open(trace_file, "r") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        events = data.get("traceEvents", [])
        for event in events:
            if not isinstance(event, dict):
                continue
            name = event.get("name", "")
            dur = event.get("dur", 0)  # microseconds
            dur_ms = dur / 1000.0
            detail = event.get("args", {}).get("detail", "")

            if name == "Source" and detail:
                header_times[detail] = header_times.get(detail, 0) + dur_ms
            elif name in ("InstantiateClass", "InstantiateFunction") and detail:
                template_times[detail] = template_times.get(detail, 0) + dur_ms
            elif name == "Total ExecuteCompiler" and detail:
                source_times[detail] = source_times.get(detail, 0) + dur_ms

    # Print header times
    if header_times:
        sorted_headers = sorted(header_times.items(), key=lambda x: x[1], reverse=True)
        print(f"\n{'='*80}")
        print(f"CLANG -ftime-trace: Top {min(top_n, len(sorted_headers))} headers by parse time")
        print(f"{'='*80}")
        print(f"{'Time (ms)':>10}  {'Header'}")
        print(f"{''*80}")
        for header, time_ms in sorted_headers[:top_n]:
            display = header
            if len(display) > 65:
                display = "..." + display[-62:]
            print(f"{time_ms:>10.1f}  {display}")
        print(f"{''*80}")

    # Print template instantiation times
    if template_times:
        sorted_templates = sorted(template_times.items(), key=lambda x: x[1], reverse=True)
        print(f"\n{'='*80}")
        print(
            f"CLANG -ftime-trace: Top {min(top_n, len(sorted_templates))} template instantiations"
        )
        print(f"{'='*80}")
        print(f"{'Time (ms)':>10}  {'Template'}")
        print(f"{''*80}")
        for template, time_ms in sorted_templates[:top_n]:
            display = template
            if len(display) > 65:
                display = "..." + display[-62:]
            print(f"{time_ms:>10.1f}  {display}")
        print(f"{''*80}")

    return True


def generate_ninja_trace(build_dir: Path, output_file: Path) -> bool:
    """
    Convert .ninja_log to Chrome trace format for visualization.

    :param build_dir: Path to the build directory containing .ninja_log.
    :param output_file: Path to write the Chrome trace JSON output.
    :return: True if the trace was generated successfully, False otherwise.
    """
    ninja_log = build_dir / ".ninja_log"
    if not ninja_log.exists():
        print(f"No .ninja_log found at {ninja_log}")
        return False

    trace_events: list[dict[str, Any]] = []
    with open(ninja_log, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            try:
                start_ms = int(parts[0])
                end_ms = int(parts[1])
                output = parts[3] if len(parts) > 3 else parts[2]
                # Use short name for display
                name = os.path.basename(output)
                trace_events.append(
                    {
                        "name": name,
                        "cat": "build",
                        "ph": "X",  # complete event
                        "ts": start_ms * 1000,  # microseconds
                        "dur": (end_ms - start_ms) * 1000,
                        "pid": 1,
                        "tid": 0,  # will be assigned below
                        "args": {"file": output},
                    }
                )
            except (ValueError, IndexError):
                continue

    if not trace_events:
        print("No entries found in .ninja_log")
        return False

    # Assign thread IDs to visualize parallelism
    # Sort by start time, then greedily assign to first available "lane"
    trace_events.sort(key=lambda e: e["ts"])
    lane_end_times: list[int] = []
    for event in trace_events:
        start = event["ts"]
        assigned = False
        for i, end_time in enumerate(lane_end_times):
            if start >= end_time:
                event["tid"] = i
                lane_end_times[i] = start + event["dur"]
                assigned = True
                break
        if not assigned:
            event["tid"] = len(lane_end_times)
            lane_end_times.append(start + event["dur"])

    trace_data = {"traceEvents": trace_events}
    with open(output_file, "w") as f:
        json.dump(trace_data, f)

    print(f"Ninja trace written to: {output_file}")
    print(f"Open in Chrome (chrome://tracing) or https://ui.perfetto.dev/ to visualize.")
    print(f"Total targets: {len(trace_events)}, max parallelism: {len(lane_end_times)}")
    return True


def find_build_dir(args: argparse.Namespace) -> Path:
    """
    Determine the build directory from arguments.

    :param args: Parsed command-line arguments.
    :return: Path to the build directory.
    """
    if args.build_dir:
        return Path(args.build_dir)
    if args.preset:
        return Path("build") / args.preset
    # Try to auto-detect
    build_path = Path("build")
    if build_path.exists():
        # Look for directories with .ninja_log
        for entry in sorted(build_path.iterdir()):
            if entry.is_dir() and (entry / ".ninja_log").exists():
                print(f"Auto-detected build directory: {entry}")
                return entry
    print("Could not determine build directory. Use --build-dir or --preset.")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze build performance metrics for SlangPy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Path to the build directory (e.g., build/windows-msvc)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="CMake preset name (used to locate build dir as build/<preset>)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top entries to display (default: 30)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to captured build output (stderr) for MSVC /Bt+ parsing",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Output path for Chrome trace JSON (converts .ninja_log to trace format)",
    )

    args = parser.parse_args()
    build_dir = find_build_dir(args)

    if not build_dir.exists():
        print(f"Build directory does not exist: {build_dir}")
        sys.exit(1)

    found_data = False

    # Generate trace if requested
    if args.trace:
        if not generate_ninja_trace(build_dir, Path(args.trace)):
            sys.exit(1)
        return

    # Always try ninja log
    if parse_ninja_log(build_dir, args.top):
        found_data = True

    # Try MSVC log if provided
    if args.log_file:
        if parse_msvc_bt(Path(args.log_file), args.top):
            found_data = True

    # Try Clang -ftime-trace files
    if parse_ftime_trace(build_dir, args.top):
        found_data = True

    if not found_data:
        print("\nNo build metrics data found.")
        print("Run a build first, or specify --log-file for MSVC output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
