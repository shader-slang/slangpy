# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from pathlib import Path

from slangpy.testing import helpers

from bench.table import display
from bench.report import BenchmarkReport, generate_report, write_report, upload_report

from typing import Any

SHADER_DIR = Path(__file__).parent


def pytest_sessionstart(session: pytest.Session):
    helpers.start_session(shader_include_paths=[SHADER_DIR])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    helpers.finish_session()

    # Generate benchmark report
    benchmark_reports: list[BenchmarkReport] = session.config._benchmark_reports  # type: ignore
    report = generate_report(benchmark_reports)

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


def pytest_runtest_setup(item: Any):
    helpers.setup_test()


def pytest_runtest_teardown(item: Any, nextitem: Any):
    helpers.teardown_test()


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


def pytest_configure(config: pytest.Config):
    # Setup list for storing benchmark reports
    config._benchmark_reports = []  # type: ignore


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int):
    benchmark_reports: list[BenchmarkReport] = terminalreporter.config._benchmark_reports  # type: ignore
    display(benchmark_reports)
