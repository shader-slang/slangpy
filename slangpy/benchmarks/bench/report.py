# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypedDict, Any
import json
from bench.utils import get_project_info, get_machine_info, get_commit_info


class BenchmarkReport(TypedDict):
    name: str
    cpu_time: float
    data: list[float]
    min: float
    max: float
    mean: float
    median: float
    stddev: float


class Report(TypedDict):
    project_info: dict[str, Any]
    machine_info: dict[str, Any]
    commit_info: dict[str, Any]
    benchmarks: list[BenchmarkReport]


def generate_report(benchmarks: list[BenchmarkReport]) -> Report:
    return {
        "project_info": get_project_info(),
        "machine_info": get_machine_info(),
        "commit_info": get_commit_info(),
        "benchmarks": benchmarks,
    }


def write_report(report: Report, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=4)


def load_report(path: str) -> Report:
    with open(path, "r") as f:
        return json.load(f)
