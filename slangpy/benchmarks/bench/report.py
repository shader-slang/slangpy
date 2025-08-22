# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypedDict
import json


class Report(TypedDict):
    name: str
    min: float
    max: float
    mean: float
    median: float
    stddev: float


def write_reports(reports: list[Report], path: str) -> None:
    with open(path, "w") as f:
        json.dump(reports, f, indent=4)


def load_reports(path: str) -> list[Report]:
    with open(path, "r") as f:
        return json.load(f)
