# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest

gpu_clock = import_module("slangpy.testing.benchmark.gpu_clock")


@pytest.fixture
def clock_commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Model one GPU without executing any commands that change its clocks."""
    commands: list[list[str]] = []
    monkeypatch.setattr(gpu_clock, "get_gpu_name", lambda _: "Test GPU")
    monkeypatch.setattr(gpu_clock, "enumerate_gpu_clocks", lambda _: [(1000, 2000)])
    monkeypatch.setattr(gpu_clock, "run_command", lambda command: (commands.append(command), "")[1])
    return commands


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_ratio_does_not_touch_gpu(monkeypatch: pytest.MonkeyPatch, ratio: float) -> None:
    def unexpected_query(_: int) -> str:
        pytest.fail("Invalid ratios must fail before querying or changing the GPU")

    monkeypatch.setattr(gpu_clock, "get_gpu_name", unexpected_query)
    with pytest.raises(ValueError, match="ratio must be between"):
        gpu_clock.lock_gpu_clocks(0, ratio, conservative=True)


def test_lock_and_unlock(clock_commands: list[list[str]]) -> None:
    assert gpu_clock.lock_gpu_clocks(2, 1.0, conservative=True) == (1000, 2000)
    gpu_clock.unlock_gpu_clocks(2)
    assert [command[-1] for command in clock_commands] == [
        "--lock-memory-clocks=1000",
        "--lock-gpu-clocks=2000",
        "--reset-memory-clocks",
        "--reset-gpu-clocks",
    ]
    assert all(command[-3:-1] == ["-i", "2"] for command in clock_commands)


def test_dry_run_does_not_change_clocks(clock_commands: list[list[str]]) -> None:
    assert gpu_clock.lock_gpu_clocks(0, 1.0, conservative=True, dry_run=True) == (1000, 2000)
    gpu_clock.unlock_gpu_clocks(0, dry_run=True)
    assert clock_commands == []


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_graphics_lock_attempts_memory_reset_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch, clock_commands: list[list[str]], cleanup_fails: bool
) -> None:
    original_error = subprocess.CalledProcessError(1, "graphics lock")

    def run(command: list[str]) -> str:
        clock_commands.append(command)
        if command[-1].startswith("--lock-gpu-clocks"):
            raise original_error
        if command[-1] == "--reset-memory-clocks" and cleanup_fails:
            raise subprocess.CalledProcessError(2, "memory reset")
        return ""

    monkeypatch.setattr(gpu_clock, "run_command", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        gpu_clock.lock_gpu_clocks(0, 1.0, conservative=True)
    assert error.value is original_error
    assert [command[-1] for command in clock_commands] == [
        "--lock-memory-clocks=1000",
        "--lock-gpu-clocks=2000",
        "--reset-memory-clocks",
    ]


@pytest.mark.parametrize("fail_memory,fail_graphics", [(True, False), (False, True), (True, True)])
def test_unlock_attempts_both_resets_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
    clock_commands: list[list[str]],
    fail_memory: bool,
    fail_graphics: bool,
) -> None:
    def run(command: list[str]) -> str:
        clock_commands.append(command)
        if (command[-1] == "--reset-memory-clocks" and fail_memory) or (
            command[-1] == "--reset-gpu-clocks" and fail_graphics
        ):
            raise subprocess.CalledProcessError(1, command)
        return ""

    monkeypatch.setattr(gpu_clock, "run_command", run)
    with pytest.raises(subprocess.CalledProcessError):
        gpu_clock.unlock_gpu_clocks(0)
    assert [command[-1] for command in clock_commands] == [
        "--reset-memory-clocks",
        "--reset-gpu-clocks",
    ]


def test_unlock_does_not_depend_on_gpu_name_query(
    monkeypatch: pytest.MonkeyPatch, clock_commands: list[list[str]]
) -> None:
    def unavailable_name(_: int) -> str:
        raise subprocess.CalledProcessError(1, "query-gpu=name")

    monkeypatch.setattr(gpu_clock, "get_gpu_name", unavailable_name)
    gpu_clock.unlock_gpu_clocks(0)
    assert len(clock_commands) == 2


def test_cli_works_without_importing_slangpy(tmp_path: Path) -> None:
    """An unavailable native extension must not prevent the cleanup CLI from starting."""
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-S", str(root / "tools/gpu_clock.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unlock" in result.stdout
