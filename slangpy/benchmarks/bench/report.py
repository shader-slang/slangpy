# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypedDict, Any
import json
from datetime import datetime

from bench.utils import get_project_info, get_machine_info, get_commit_info, to_json, from_json


class BenchmarkReport(TypedDict):
    name: str
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
    project_info: dict[str, Any]
    machine_info: dict[str, Any]
    commit_info: dict[str, Any]
    benchmarks: list[BenchmarkReport]


def generate_report(benchmarks: list[BenchmarkReport]) -> Report:
    return {
        "timestamp": datetime.now(),
        "project_info": get_project_info(),
        "machine_info": get_machine_info(),
        "commit_info": get_commit_info(),
        "benchmarks": benchmarks,
    }


def write_report(report: Report, path: str) -> None:
    with open(path, "w") as f:
        json.dump(to_json(report), f, indent=4)


def load_report(path: str) -> Report:
    with open(path, "r") as f:
        return from_json(json.load(f))


def upload_report(report: Report, connection_string: str, database_name: str):
    from pymongo import MongoClient

    client = MongoClient(connection_string)
    db = client[database_name]
    db["benchmark"].insert_one(report)
