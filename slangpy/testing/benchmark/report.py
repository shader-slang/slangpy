# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datetime import datetime
import json
from pathlib import Path
from typing import Any, TypedDict

from .utils import from_json, get_commit_info, get_machine_info, get_project_info, to_json


class BenchmarkReport(TypedDict):
    name: str
    filename: str
    function: str
    params: dict[str, str]
    meta: dict[str, str]
    timestamp: datetime
    cpu_time: float
    data: list[float]
    min: float
    max: float
    mean: float
    median: float
    stddev: float


class Report(TypedDict):
    timestamp: datetime
    run_id: str
    project_info: dict[str, Any]
    machine_info: dict[str, Any]
    commit_info: dict[str, Any]
    benchmarks: list[BenchmarkReport]


def generate_report(timestamp: datetime, run_id: str, benchmarks: list[BenchmarkReport]) -> Report:
    """Collect local project, machine, commit, and benchmark data into a legacy report."""

    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "project_info": get_project_info(),
        "machine_info": get_machine_info(),
        "commit_info": get_commit_info(),
        "benchmarks": benchmarks,
    }


def generate_run_id(report: Report) -> str:
    """Derive a readable local report ID from its timestamp and commit state."""

    timestamp = report["timestamp"].strftime("%Y%m%d-%H%M%S")
    commit_id = report["commit_info"].get("id", "unknown")
    commit_dirty = "dirty" if report["commit_info"].get("dirty", False) else "clean"
    return f"{timestamp}-{commit_id}-{commit_dirty}"


def strip_benchmark_data(report: Report) -> Report:
    """Copy a local report while removing its raw benchmark sample arrays."""

    stripped_benchmarks = []
    for benchmark in report["benchmarks"]:
        stripped_benchmark = benchmark.copy()
        stripped_benchmark["data"] = []
        stripped_benchmarks.append(stripped_benchmark)
    stripped_report = report.copy()
    stripped_report["benchmarks"] = stripped_benchmarks
    return stripped_report


def write_report(report: Report, path: Path, strip_data: bool = False) -> None:
    """Serialize a legacy local report, optionally omitting its raw samples."""

    if strip_data:
        report = strip_benchmark_data(report)
    with open(path, "w") as f:
        json.dump(to_json(report), f, indent=4)


def load_report(path: Path) -> Report:
    """Load a legacy local report from a JSON file."""

    with open(path, "r") as f:
        return from_json(json.load(f))


def list_report_ids(dir: Path) -> list[str]:
    """List saved local report IDs from newest to oldest."""

    files = list(dir.iterdir())
    # Sort reports by file date in descending order.
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return [f.stem for f in files if f.suffix == ".json"]
