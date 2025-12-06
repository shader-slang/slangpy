# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from datetime import datetime
from pathlib import Path

from .report import (
    Report,
    BenchmarkReport,
    generate_report,
    generate_run_id,
    list_report_ids,
    write_report,
    load_report,
    upload_report,
)
from .table import display

from typing import Any, TypedDict, Optional

BENCHMARK_DIR = Path(".benchmarks")


class Context(TypedDict):
    timestamp: datetime
    benchmark_reports: list[BenchmarkReport]
    compare_run_id: Optional[str]


def get_context(config: pytest.Config) -> Context:
    if not hasattr(config, "_benchmark_context"):
        context: Context = {
            "timestamp": datetime.now(),
            "benchmark_reports": [],
            "compare_run_id": None,
        }
        setattr(config, "_benchmark_context", context)
    return getattr(config, "_benchmark_context")


def pytest_configure(config: pytest.Config):
    # Make sure context is initialized
    get_context(config)


def pytest_sessionstart(session: pytest.Session):
    get_context(session.config)["timestamp"] = datetime.now()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    # Generate benchmark report
    context = get_context(session.config)
    report = generate_report(context["timestamp"], "", context["benchmark_reports"])

    # Save report
    save = session.config.getoption("--benchmark-save")
    if save != "_unspecified_":
        run_id = save if save else generate_run_id(report)
        report["run_id"] = run_id
        path = BENCHMARK_DIR / (run_id + ".json")
        print(f"Saving benchmark report to {path}")
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        write_report(report, path, strip_data=True)

    # Upload report to MongoDB
    upload = session.config.getoption("--benchmark-upload")
    if upload:
        report["run_id"] = upload
        print("Uploading benchmark report to MongoDB")
        connection_string = session.config.getoption("--benchmark-mongodb-connection-string")
        database_name = session.config.getoption("--benchmark-mongodb-database-name")
        upload_report(report, connection_string, database_name)


def pytest_addoption(parser: pytest.Parser):
    group = parser.getgroup("benchmarking")
    group.addoption(
        "--benchmark-save",
        action="store",
        default="_unspecified_",
        nargs="?",
        metavar="ID",
        help="Save the current benchmark run to a file. Optionally specify a run ID.",
    )
    group.addoption(
        "--benchmark-compare",
        action="store",
        default="_unspecified_",
        nargs="?",
        metavar="ID",
        help="Compare against previously saved benchmark run. Optionally specify a run ID. By default, use the latest run.",
    )
    group.addoption(
        "--benchmark-list-runs",
        action="store_true",
        default=False,
        help="List the IDs of all saved benchmark runs.",
    )
    group.addoption(
        "--benchmark-upload",
        action="store",
        default=False,
        metavar="ID",
        help="Upload benchmark report to a MongoDB with the specified run ID.",
    )
    group.addoption(
        "--benchmark-mongodb-connection-string",
        action="store",
        default="mongodb://localhost:27017",
        metavar="CONNECTION_STRING",
        help="MongoDB connection string.",
    )
    group.addoption(
        "--benchmark-mongodb-database-name",
        action="store",
        default="nvr-ci",
        metavar="NAME",
        help="MongoDB database name.",
    )


def pytest_cmdline_main(config: pytest.Config):
    compare = config.getoption("--benchmark-compare")
    if compare != "_unspecified_":
        ids = list_report_ids(BENCHMARK_DIR)
        if len(ids) == 0:
            print("No benchmark runs found!")
            return 1
        id = compare if compare else ids[0]
        if not id in ids:
            print(f'Benchmark run "{id}" not found!')
            return 1
        print(f"Comparing against benchmark run: {id}")
        get_context(config)["compare_run_id"] = id

    if config.getoption("--benchmark-list-runs"):
        print("Benchmark runs:")
        ids = list_report_ids(BENCHMARK_DIR)
        for id in ids:
            print(id)
        return 0


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int):
    context = get_context(terminalreporter.config)
    benchmark_reports: list[BenchmarkReport] = context["benchmark_reports"]
    baseline_report: Optional[Report] = None
    if context["compare_run_id"]:
        baseline_report = load_report(BENCHMARK_DIR / (context["compare_run_id"] + ".json"))
    display(
        terminalreporter.config.get_terminal_writer(),
        benchmark_reports,
        baseline_benchmarks=baseline_report["benchmarks"] if baseline_report else None,
    )
