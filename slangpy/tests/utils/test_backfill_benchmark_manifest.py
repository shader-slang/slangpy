# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
from pathlib import Path

import pytest

from slangpy.testing.benchmark import plugin
from tools.backfill_benchmark_manifest import (
    benchmark_modules,
    benchmarks_postdating,
    exists_in_commit,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository where one benchmark exists from the start and one is added later."""

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
    return root


def test_benchmark_modules_lists_working_tree(repo: Path) -> None:
    assert benchmark_modules(repo, "slangpy/benchmarks") == [
        "slangpy/benchmarks/test_benchmark_new.py",
        "slangpy/benchmarks/test_benchmark_old.py",
    ]


def test_benchmark_modules_tolerates_missing_directory(repo: Path) -> None:
    assert benchmark_modules(repo, "does/not/exist") == []


def test_exists_in_commit_distinguishes_history(repo: Path) -> None:
    first = git(repo, "rev-list", "--max-parents=0", "HEAD")
    assert exists_in_commit(repo, first, "slangpy/benchmarks/test_benchmark_old.py")
    assert not exists_in_commit(repo, first, "slangpy/benchmarks/test_benchmark_new.py")


def test_only_benchmarks_the_target_predates_are_listed(repo: Path) -> None:
    first = git(repo, "rev-list", "--max-parents=0", "HEAD")
    assert benchmarks_postdating(repo, first, "slangpy/benchmarks") == [
        "slangpy/benchmarks/test_benchmark_new.py"
    ]


def test_nothing_is_listed_for_the_newest_target(repo: Path) -> None:
    assert benchmarks_postdating(repo, "HEAD", "slangpy/benchmarks") == []


class FakeConfig:
    def __init__(self, rootpath: Path) -> None:
        self.rootpath = rootpath


class FakeItem:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.markers: list[pytest.MarkDecorator] = []

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.markers.append(marker)


def collect(
    monkeypatch: pytest.MonkeyPatch, root: Path, listed: set[str], item_paths: list[str]
) -> list[FakeItem]:
    """Run the collection hook with a given manifest and return the items."""

    monkeypatch.setattr(plugin, "BENCHMARKS_POSTDATING_TARGET", frozenset(listed))
    items = [FakeItem(root / p) for p in item_paths]
    plugin.pytest_collection_modifyitems(FakeConfig(root), items)  # type: ignore[arg-type]
    return items


def test_listed_benchmarks_are_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    new, old = collect(
        monkeypatch,
        tmp_path,
        {"slangpy/benchmarks/test_benchmark_new.py"},
        ["slangpy/benchmarks/test_benchmark_new.py", "slangpy/benchmarks/test_benchmark_old.py"],
    )
    assert [m.kwargs["reason"] for m in new.markers] == [
        f"slangpy/benchmarks/test_benchmark_new.py postdates backfill target "
        f"{plugin.BACKFILL_TARGET_SHA[:12]}"
    ]
    assert old.markers == []


def test_an_empty_manifest_skips_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (item,) = collect(monkeypatch, tmp_path, set(), ["slangpy/benchmarks/test_benchmark_new.py"])
    assert item.markers == []


def test_items_outside_the_root_are_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        plugin, "BENCHMARKS_POSTDATING_TARGET", frozenset({"slangpy/benchmarks/test_x.py"})
    )
    item = FakeItem(Path("/elsewhere/slangpy/benchmarks/test_x.py"))
    plugin.pytest_collection_modifyitems(FakeConfig(tmp_path), [item])  # type: ignore[arg-type]
    assert item.markers == []
