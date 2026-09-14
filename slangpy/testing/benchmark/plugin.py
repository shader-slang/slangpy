# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from datetime import datetime
import os
from pathlib import Path
from uuid import uuid4

from .benchview import (
    BenchViewObservation,
    BenchmarkSubmissionError,
    benchview_submission_url,
    build_benchview_submissions,
    submit_benchview_submissions,
)
from .report import (
    BenchmarkReport,
    Report,
    generate_report,
    generate_run_id,
    list_report_ids,
    load_report,
    write_report,
)
from .table import display

from typing import Any, TypedDict, Optional

BENCHMARK_DIR = Path(".benchmarks")

# The historical backfill runs today's benchmarks against year-old builds, so a
# benchmark shader may call library APIs that did not exist yet. Without this,
# one such benchmark fails the whole device and discards that commit's results.
# Only the backfill workflow sets BACKFILL_TARGET_SHA; ordinary CI still fails.
BACKFILL_TARGET_SHA = os.environ.get("BACKFILL_TARGET_SHA", "")

# A benchmark module that exists in the target build can still contain newer tests
# whose shaders the target's Slang cannot compile. Whole modules that postdate the
# target are handled up front by the manifest below, not here.
INCOMPATIBLE_TARGET_EXCEPTIONS = ("SlangCompileError",)


def _benchmarks_postdating_target() -> frozenset[str]:
    """Benchmark modules the backfill target predates, as repo-relative POSIX paths.

    :return: Paths listed by tools/backfill_benchmark_manifest.py, empty when unset.
    """

    manifest = os.environ.get("BACKFILL_SKIP_BENCHMARKS", "")
    if not manifest or not Path(manifest).is_file():
        return frozenset()
    return frozenset(
        line.strip() for line in Path(manifest).read_text(encoding="utf-8").splitlines()
    ) - {""}


BENCHMARKS_POSTDATING_TARGET = _benchmarks_postdating_target()


class Context(TypedDict):
    timestamp: datetime
    benchmark_reports: list[BenchmarkReport]
    benchmark_observations: list[BenchViewObservation]
    compare_run_id: Optional[str]
    execution_id: str
    incompatible_skips: int


def get_context(config: pytest.Config) -> Context:
    if not hasattr(config, "_benchmark_context"):
        context: Context = {
            "timestamp": datetime.now(),
            "benchmark_reports": [],
            "benchmark_observations": [],
            "compare_run_id": None,
            "execution_id": str(uuid4()),
            "incompatible_skips": 0,
        }
        setattr(config, "_benchmark_context", context)
    return getattr(config, "_benchmark_context")


def apply_benchmark_source_override(report: Report) -> None:
    """Replace Git metadata when CI overlays the current reporter on historical source.

    :param report: Mutable local report whose commit facts feed native submissions.
    """

    revision = os.environ.get("BENCHVIEW_BENCHMARK_REF")
    if not revision:
        return
    report["commit_info"]["id"] = revision
    report["commit_info"]["dirty"] = False
    branch = os.environ.get("BENCHVIEW_BENCHMARK_BRANCH")
    if branch:
        report["commit_info"]["branch"] = branch


def pytest_configure(config: pytest.Config):
    # Make sure context is initialized
    get_context(config)
    submit = config.getoption("benchmark_submit")
    if submit:
        api_url = config.getoption("benchmark_api_url") or os.environ.get("BENCHVIEW_API_URL")
        if not api_url:
            raise pytest.UsageError(
                "--benchmark-submit requires --benchmark-api-url or BENCHVIEW_API_URL."
            )
        try:
            benchview_submission_url(api_url)
        except BenchmarkSubmissionError as error:
            raise pytest.UsageError(str(error)) from error
        if not os.environ.get("BENCHVIEW_API_KEY"):
            raise pytest.UsageError("--benchmark-submit requires BENCHVIEW_API_KEY.")


def pytest_sessionstart(session: pytest.Session):
    get_context(session.config)["timestamp"] = datetime.now()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip benchmarks that did not exist in the backfill target."""

    if not BENCHMARKS_POSTDATING_TARGET:
        return
    root = Path(str(config.rootpath))
    for item in items:
        try:
            relative = Path(str(item.path)).relative_to(root).as_posix()
        except ValueError:
            continue
        if relative in BENCHMARKS_POSTDATING_TARGET:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"{relative} postdates backfill target {BACKFILL_TARGET_SHA[:12]}"
                )
            )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """Report a benchmark that a backfill target cannot run as skipped."""

    outcome = yield
    if not BACKFILL_TARGET_SHA:
        return
    report = outcome.get_result()
    if not report.failed or call.excinfo is None:
        return
    exception_name = type(call.excinfo.value).__name__
    if exception_name not in INCOMPATIBLE_TARGET_EXCEPTIONS:
        return
    report.outcome = "skipped"
    report.longrepr = (
        str(item.path),
        None,
        f"Benchmark raised {exception_name} against backfill target "
        f"{BACKFILL_TARGET_SHA[:12]}; it postdates that build.",
    )
    get_context(item.config)["incompatible_skips"] += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    # Generate benchmark report
    context = get_context(session.config)

    # A device type the target build cannot create leaves a real hole in the series,
    # so say so loudly. It is not a failure: the other device types still report.
    from slangpy.testing.helpers import BACKFILL_UNAVAILABLE_DEVICES

    for reason in BACKFILL_UNAVAILABLE_DEVICES.values():
        print(f"Backfill device unavailable, benchmarks skipped: {reason}")

    # Skipping the odd incompatible benchmark still yields a usable data point, but
    # skipping every one of them does not: that is a silently empty result, so fail.
    if context["incompatible_skips"] and not context["benchmark_observations"]:
        print(
            f"All {context['incompatible_skips']} benchmark(s) were skipped as "
            f"incompatible with backfill target {BACKFILL_TARGET_SHA[:12]}; "
            "no measurements were produced."
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    report = generate_report(context["timestamp"], "", context["benchmark_reports"])
    apply_benchmark_source_override(report)

    # Save report
    save = session.config.getoption("--benchmark-save")
    if save != "_unspecified_":
        run_id = save if save else generate_run_id(report)
        report["run_id"] = run_id
        path = BENCHMARK_DIR / (run_id + ".json")
        print(f"Saving benchmark report to {path}")
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        write_report(report, path, strip_data=True)

    # Submit native observations through the authenticated BenchView HTTP boundary.
    submit = session.config.getoption("benchmark_submit")
    if submit:
        api_url = session.config.getoption("benchmark_api_url") or os.environ["BENCHVIEW_API_URL"]
        write_key = os.environ["BENCHVIEW_API_KEY"]
        submissions = build_benchview_submissions(
            context["benchmark_observations"],
            request_id=submit,
            execution_id=context["execution_id"],
            project_info=report["project_info"],
            machine_info=report["machine_info"],
            commit_info=report["commit_info"],
        )
        print(f"Submitting {len(submissions)} benchmark batch(es) to BenchView at {api_url}")
        receipts = submit_benchview_submissions(api_url, write_key, submissions)
        duplicate_count = sum(1 for receipt in receipts if receipt["duplicate"])
        print(
            f"BenchView accepted {len(receipts) - duplicate_count} batch(es); "
            f"{duplicate_count} exact retry batch(es)."
        )


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
        "--benchmark-submit",
        "--benchmark-upload",
        dest="benchmark_submit",
        action="store",
        default=False,
        metavar="REQUEST_ID",
        help="Submit native observations to BenchView with the specified request ID.",
    )
    group.addoption(
        "--benchmark-api-url",
        dest="benchmark_api_url",
        action="store",
        default=None,
        metavar="URL",
        help="BenchView root or nested base URL; defaults to BENCHVIEW_API_URL.",
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
