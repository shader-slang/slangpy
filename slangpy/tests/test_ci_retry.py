# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unit tests for the two-stage pytest retry in tools/ci.py (issue #829).

Two layers:
  * stub tests assert the retry control flow (which commands run, in what order)
    by replacing run_command;
  * integration tests drive run_pytest_with_retry against a real throwaway pytest
    suite to prove the end-to-end invariants that matter for the false-green risk
    of slang#11911 - a flaky test that passes on the sequential rerun turns the
    run green, while a deterministic failure stays red on both attempts.

The nested throwaway suites are self-contained and create no GPU device. (The
file itself is collected under slangpy/tests/, whose conftest.py imports the
native extension like any other test in that directory.)
"""

import argparse
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import pytest

_CI_PY = Path(__file__).resolve().parents[2] / "tools" / "ci.py"


def _load_ci() -> ModuleType:
    # tools/ has no __init__.py, so load ci.py directly by path.
    spec = importlib.util.spec_from_file_location("slangpy_ci_under_test", _CI_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]
    return module


def _args(parallel: bool, disable_torch: bool = False) -> argparse.Namespace:
    return argparse.Namespace(parallel=parallel, disable_torch=disable_torch, preset=None)


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(cmd)


# --------------------------------------------------------------------------- #
# Control-flow tests (run_command stubbed)
# --------------------------------------------------------------------------- #


class _RunRecorder:
    """Stub for ci.run_command: records each command and raises on the first
    `fail_first` calls to mimic a nonzero pytest exit."""

    def __init__(self, fail_first: int = 0):
        super().__init__()
        self._fail_first = fail_first
        self.calls: list[list[str]] = []

    def __call__(self, command: Any, shell: bool = True, env: Any = None) -> str:
        cmd = command if isinstance(command, list) else [command]
        self.calls.append([str(c) for c in cmd])
        if len(self.calls) <= self._fail_first:
            raise RuntimeError(f"stubbed failure for call {len(self.calls)}")
        return ""


@pytest.fixture
def ci(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    module = _load_ci()
    monkeypatch.chdir(tmp_path)  # unit_test_python does os.makedirs("reports")
    return module


def test_parallel_failure_retries_failed_tests_sequentially(
    ci: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    recorder = _RunRecorder(fail_first=1)
    monkeypatch.setattr(ci, "run_command", recorder)

    ci.unit_test_python(_args(parallel=True))

    assert len(recorder.calls) == 2, "parallel failure should trigger exactly one rerun"
    first, rerun = recorder.calls
    assert "-n auto" in _cmd_str(first) and "--maxprocesses=4" in _cmd_str(first)
    assert "--cache-clear" in first
    assert "-n 0" in _cmd_str(rerun) and "--lf" in rerun


def test_parallel_success_does_not_rerun(ci: ModuleType, monkeypatch: pytest.MonkeyPatch):
    recorder = _RunRecorder(fail_first=0)
    monkeypatch.setattr(ci, "run_command", recorder)

    ci.unit_test_python(_args(parallel=True))

    assert len(recorder.calls) == 1, "a passing first attempt must not rerun"
    assert "--lf" not in recorder.calls[0]


def test_parallel_persistent_failure_propagates(ci: ModuleType, monkeypatch: pytest.MonkeyPatch):
    recorder = _RunRecorder(fail_first=2)
    monkeypatch.setattr(ci, "run_command", recorder)

    with pytest.raises(RuntimeError):
        ci.unit_test_python(_args(parallel=True))

    assert len(recorder.calls) == 2
    assert "--lf" in recorder.calls[1]


def test_non_parallel_never_retries(ci: ModuleType, monkeypatch: pytest.MonkeyPatch):
    recorder = _RunRecorder(fail_first=1)
    monkeypatch.setattr(ci, "run_command", recorder)

    with pytest.raises(RuntimeError):
        ci.unit_test_python(_args(parallel=False))

    assert len(recorder.calls) == 1, "non-parallel run must not retry"
    only = recorder.calls[0]
    assert "--lf" not in only and "--cache-clear" not in only and "-n" not in only


def test_examples_parallel_failure_retries(ci: ModuleType, monkeypatch: pytest.MonkeyPatch):
    recorder = _RunRecorder(fail_first=1)
    monkeypatch.setattr(ci, "run_command", recorder)

    ci.test_examples(_args(parallel=True))

    assert len(recorder.calls) == 2
    assert "samples/tests" in _cmd_str(recorder.calls[0])
    assert "--lf" in recorder.calls[1]


# --------------------------------------------------------------------------- #
# Integration tests (real pytest subprocess, no stub) - prove the false-green
# guard end to end. These require pytest-xdist, which CI always has.
# --------------------------------------------------------------------------- #

_HAVE_XDIST = importlib.util.find_spec("xdist") is not None
_needs_xdist = pytest.mark.skipif(not _HAVE_XDIST, reason="pytest-xdist not installed")

# A test whose first, parallel run fails and whose sequential rerun passes once a
# marker file exists - a controllable stand-in for xdist/GPU-contention flakiness.
_FLAKY_SUITE = textwrap.dedent(
    """
    import os

    def test_recovers():
        flag = os.environ["CI829_FLAG"]
        if not os.path.exists(flag):
            open(flag, "w").close()  # arm: fail once, pass on the rerun
            raise AssertionError("simulated flake on first attempt")

    def test_always_passes():
        assert True
    """
)

_DETERMINISTIC_FAIL_SUITE = textwrap.dedent(
    """
    def test_broken():
        assert False, "a real, deterministic defect"

    def test_ok():
        assert True
    """
)


def _make_suite(tmp_path: Path) -> Path:
    """Create an isolated throwaway pytest suite. A suite-local pyproject.toml
    pins the nested run's rootdir here so its .pytest_cache stays suite-local and
    cannot collide with the outer test session's cache."""
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    return suite


def _run_suite_via_ci(
    ci: ModuleType,
    suite_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_env: Optional[dict[str, str]] = None,
) -> None:
    """Invoke run_pytest_with_retry on a real suite in parallel mode. The nested
    pytest run gets its own basetemp (a sibling of the suite, never a parent of
    it) so it does not clear the outer session's basetemp, under which our own
    tmp_path lives."""
    monkeypatch.chdir(suite_dir)
    monkeypatch.setattr(ci, "PYTEST_BASE_TEMP_DIR", tmp_path / "nested-basetemp")
    env = {"PYTHONPATH": str(suite_dir)}
    if extra_env:
        env.update(extra_env)
    ci.run_pytest_with_retry(str(suite_dir), env, parallel=True)


@_needs_xdist
def test_integration_flake_recovers_green(
    ci: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    suite = _make_suite(tmp_path)
    (suite / "test_flaky.py").write_text(_FLAKY_SUITE)
    monkeypatch.setenv("CI829_FLAG", str(tmp_path / "flag"))

    # First (parallel) attempt fails on the flake; the sequential --lf rerun
    # re-runs only that test, which now passes -> run_pytest_with_retry returns
    # without raising.
    _run_suite_via_ci(ci, suite, tmp_path, monkeypatch)


@_needs_xdist
def test_integration_real_failure_stays_red(
    ci: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    suite = _make_suite(tmp_path)
    (suite / "test_broken.py").write_text(_DETERMINISTIC_FAIL_SUITE)

    # The deterministic failure fails on both the parallel attempt and the
    # sequential rerun, so the RuntimeError from the rerun must propagate - the
    # slang#11911 masking case must not occur.
    with pytest.raises(RuntimeError):
        _run_suite_via_ci(ci, suite, tmp_path, monkeypatch)


# The stale-cache masking risk only bites when the first attempt dies before
# pytest updates the last-failed record. pytest merges into the existing
# last-failed mapping rather than replacing it, so a stale entry for a test that
# was not collected this run can survive a normal completion - only --cache-clear
# removes it up front. This conftest makes the first parallel attempt hard-exit at
# session start (before any cache write) so a pre-seeded stale record stays in
# place: exactly the crash path where --cache-clear earns its keep.
_CRASH_ONCE_CONFTEST = textwrap.dedent(
    """
    import os
    def pytest_sessionstart(session):
        flag = os.environ["CI829_CRASH_FLAG"]
        if not os.path.exists(flag):
            open(flag, "w").close()
            os._exit(70)  # die before tests run or the cache is written
    """
)


@_needs_xdist
def test_integration_stale_cache_cannot_mask_failure(
    ci: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    suite = _make_suite(tmp_path)
    (suite / "test_broken.py").write_text(_DETERMINISTIC_FAIL_SUITE)
    (suite / "conftest.py").write_text(_CRASH_ONCE_CONFTEST)
    monkeypatch.setenv("CI829_CRASH_FLAG", str(tmp_path / "crash_flag"))

    # Pre-seed a stale last-failed record naming a test that now passes. The
    # first attempt crashes without updating it; only --cache-clear discards it,
    # so a --lf rerun then re-runs the full suite and the real defect keeps the
    # run red. Without --cache-clear the rerun would select only the stale,
    # now-passing test and exit 0 - the slang#11911 masking this guards against.
    cache = suite / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "lastfailed").write_text('{\n  "test_broken.py::test_ok": true\n}\n')

    with pytest.raises(RuntimeError):
        _run_suite_via_ci(ci, suite, tmp_path, monkeypatch)


# Records each execution of the failing test to a file so a test can assert the
# rerun actually re-ran it, rather than inferring from the exit code alone.
_RECORDING_FAIL_SUITE = textwrap.dedent(
    """
    import os

    def test_broken():
        with open(os.environ["CI829_RAN_MARKER"], "a") as f:
            f.write("x")
        assert False, "a real, deterministic defect"

    def test_ok():
        assert True
    """
)


@_needs_xdist
def test_integration_crash_rerun_reexecutes_failing_test(
    ci: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    suite = _make_suite(tmp_path)
    (suite / "test_broken.py").write_text(_RECORDING_FAIL_SUITE)
    (suite / "conftest.py").write_text(_CRASH_ONCE_CONFTEST)
    monkeypatch.setenv("CI829_CRASH_FLAG", str(tmp_path / "crash_flag"))
    ran_marker = tmp_path / "ran_marker"
    monkeypatch.setenv("CI829_RAN_MARKER", str(ran_marker))

    # The first attempt crashes before running any test or writing the cache, so
    # the rerun starts with an empty last-failed set. --lfnf=all must then make the
    # rerun re-run the whole suite, so the real failure is actually re-executed and
    # the run stays red - rather than an empty --lf selection that skips it. An
    # inherited PYTEST_ADDOPTS=--lfnf=none (which would deselect everything) must
    # not defeat this. The re-run marker proves the failing test truly re-ran, so
    # the guarantee does not rest on pytest's exit code for an empty selection.
    with pytest.raises(RuntimeError):
        _run_suite_via_ci(
            ci, suite, tmp_path, monkeypatch, extra_env={"PYTEST_ADDOPTS": "--lfnf=none"}
        )

    # The failing test ran exactly once - on the rerun (the first attempt crashed
    # before executing anything). An empty --lf selection would leave this unwritten.
    assert ran_marker.read_text() == "x"
