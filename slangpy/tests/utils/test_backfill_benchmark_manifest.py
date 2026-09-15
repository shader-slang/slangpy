# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
from pathlib import Path

import pytest

from slangpy.testing.benchmark import plugin
from tools.backfill_benchmark_manifest import benchmarks_postdating


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_only_benchmarks_the_target_predates_are_listed(tmp_path: Path) -> None:
    """A benchmark absent from the target's tree is listed; one present in it is not."""

    root = tmp_path / "repo"
    (root / "slangpy" / "benchmarks").mkdir(parents=True)
    git(tmp_path, "init", "-q", "repo")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / "slangpy" / "benchmarks" / "test_benchmark_old.py").write_text("", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "old benchmark")
    (root / "slangpy" / "benchmarks" / "test_benchmark_new.py").write_text("", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "new benchmark")

    first = git(root, "rev-list", "--max-parents=0", "HEAD")
    assert benchmarks_postdating(root, first, "slangpy/benchmarks") == [
        "slangpy/benchmarks/test_benchmark_new.py"
    ]
    assert benchmarks_postdating(root, "HEAD", "slangpy/benchmarks") == []


class FakeConfig:
    def __init__(self, rootpath: Path) -> None:
        self.rootpath = rootpath


class FakeItem:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.markers: list[pytest.MarkDecorator] = []

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.markers.append(marker)


def test_listed_benchmarks_are_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only the listed module is skipped, and only when it resolves under the root."""

    monkeypatch.setattr(
        plugin,
        "BENCHMARKS_POSTDATING_TARGET",
        frozenset({"slangpy/benchmarks/test_benchmark_new.py"}),
    )
    listed = FakeItem(tmp_path / "slangpy/benchmarks/test_benchmark_new.py")
    present = FakeItem(tmp_path / "slangpy/benchmarks/test_benchmark_old.py")
    # An item outside the root cannot be made relative to it, so it is left alone.
    elsewhere = FakeItem(Path("/elsewhere/slangpy/benchmarks/test_benchmark_new.py"))

    plugin.pytest_collection_modifyitems(  # type: ignore[arg-type]
        FakeConfig(tmp_path), [listed, present, elsewhere]
    )

    assert [m.kwargs["reason"] for m in listed.markers] == [
        f"slangpy/benchmarks/test_benchmark_new.py postdates backfill target "
        f"{plugin.BACKFILL_TARGET_SHA[:12]}"
    ]
    assert present.markers == []
    assert elsewhere.markers == []
