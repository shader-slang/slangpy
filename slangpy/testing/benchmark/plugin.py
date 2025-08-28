# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from datetime import datetime

from .report import BenchmarkReport, generate_report, write_report, upload_report
from .table import display

from typing import Any, TypedDict


class Context(TypedDict):
    timestamp: datetime
    benchmark_reports: list[BenchmarkReport]


def get_context(config: pytest.Config) -> Context:
    return config._benchmark_context  # type: ignore


def pytest_configure(config: pytest.Config):
    context: Context = {
        "timestamp": datetime.now(),
        "benchmark_reports": [],
    }
    config._benchmark_context = context  # type: ignore


def pytest_sessionstart(session: pytest.Session):
    get_context(session.config)["timestamp"] = datetime.now()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    # Generate benchmark report
    context = get_context(session.config)
    report = generate_report(context["timestamp"], context["benchmark_reports"])

    # Write report to JSON
    if session.config.getoption("--write-benchmark-report"):
        path = session.config.getoption("--benchmark-report-path")
        print(f"Writing benchmark report to {path}")
        write_report(report, path)

    # Upload report to MongoDB
    if session.config.getoption("--upload-benchmark-report"):
        print("Uploading benchmark report to MongoDB")
        connection_string = session.config.getoption("--mongodb-connection-string")
        database_name = session.config.getoption("--mongodb-database-name")
        upload_report(report, connection_string, database_name)


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--write-benchmark-report",
        action="store_true",
        default=False,
        help="Write benchmark report to a JSON file",
    )
    parser.addoption(
        "--benchmark-report-path",
        action="store",
        default="benchmark_report.json",
        help="Path to the benchmark report JSON file",
    )
    parser.addoption(
        "--upload-benchmark-report",
        action="store_true",
        default=False,
        help="Upload benchmark report to a MongoDB",
    )
    parser.addoption(
        "--mongodb-connection-string",
        action="store",
        default="mongodb://localhost:27017",
        help="MongoDB connection string",
    )
    parser.addoption(
        "--mongodb-database-name",
        action="store",
        default="nvr-ci",
        help="MongoDB database name",
    )


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int):
    context = get_context(terminalreporter.config)
    benchmark_reports: list[BenchmarkReport] = context["benchmark_reports"]
    display(benchmark_reports)
