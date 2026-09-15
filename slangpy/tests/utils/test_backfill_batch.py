# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Offline coverage for batching a rev list into a single backfill job."""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from tools import backfill_batch as batch
from tools import benchmark_actions as actions
from tools import dispatch_backfill_batches as dispatcher


def commits(count: int) -> list[actions.Commit]:
    """Create ``count`` commits one day apart, oldest first."""

    start = datetime(2025, 9, 2, tzinfo=timezone.utc)
    return [
        actions.Commit(
            sha=f"{index:040x}",
            committed_at=start + timedelta(days=index),
            message=f"commit {index}",
            html_url=f"https://github.test/commit/{index:040x}",
        )
        for index in range(count)
    ]


def test_rev_list_accepts_the_separators_a_workflow_input_may_carry():
    assert batch.parse_rev_list("aaa bbb") == ["aaa", "bbb"]
    assert batch.parse_rev_list("aaa,bbb") == ["aaa", "bbb"]
    assert batch.parse_rev_list(" aaa,\n bbb\t ccc ") == ["aaa", "bbb", "ccc"]


def test_rev_list_benchmarks_a_repeated_commit_once():
    assert batch.parse_rev_list("aaa bbb aaa") == ["aaa", "bbb"]


def test_a_failing_commit_does_not_abort_the_rest_of_the_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    attempted: list[str] = []

    def fake_benchmark(sha: str, **_: Any) -> None:
        attempted.append(sha)
        if sha == "bbb":
            raise batch.CommitFailure("build", "exited with code 1")

    monkeypatch.setattr(batch, "benchmark_commit", fake_benchmark)
    monkeypatch.setattr(batch, "git_output", lambda *_: "vcpkgsha")

    summary = tmp_path / "summary.md"
    code = batch.run_batch(
        ["aaa", "bbb", "ccc"],
        workspace=tmp_path,
        clone=tmp_path,
        skip_manifest=tmp_path / "skip.txt",
        run_id="1",
        api_url="https://benchview.test",
        summary_path=summary,
    )

    assert attempted == ["aaa", "bbb", "ccc"], "the batch stopped at the failing commit"
    assert code == 0, "a single failure must not fail the whole job"
    report = summary.read_text(encoding="utf-8")
    assert "**2 succeeded, 1 failed, 0 not reached.**" in report
    assert "Failed: `bbb`" in report


def test_the_summary_is_rewritten_as_the_batch_proceeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A job killed mid-batch must still say which commits it reached."""

    summary = tmp_path / "summary.md"
    seen: list[str] = []

    def fake_benchmark(sha: str, **_: Any) -> None:
        if sha == "ccc":
            # Stand in for the job being killed: the summary written so far is all
            # that survives, and it has to name the commits that never ran.
            seen.append(summary.read_text(encoding="utf-8"))
            raise KeyboardInterrupt

    monkeypatch.setattr(batch, "benchmark_commit", fake_benchmark)
    monkeypatch.setattr(batch, "git_output", lambda *_: "vcpkgsha")

    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(
            ["aaa", "bbb", "ccc", "ddd"],
            workspace=tmp_path,
            clone=tmp_path,
            skip_manifest=tmp_path / "skip.txt",
            run_id="1",
            api_url="https://benchview.test",
            summary_path=summary,
        )

    partial = seen[0]
    assert "**2 succeeded, 0 failed, 2 not reached.**" in partial
    assert "Not reached: `ccc ddd`" in partial


def test_the_report_names_every_commit_exactly_once():
    outcomes = [
        batch.CommitOutcome("aaa", "success", None, 60.0),
        batch.CommitOutcome("bbb", "failed", "benchmark", 30.0),
    ]
    report = batch.format_report(outcomes, ["ccc"])
    for sha in ("aaa", "bbb", "ccc"):
        assert report.count(f"`{sha}`") == 2, f"{sha} is not both tabulated and listed"


def record_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
    """Run one commit with every subprocess stubbed, returning the commands issued."""

    issued: list[list[str]] = []

    def fake_run(command, cwd, stage, env=None):
        issued.append(list(command))

    def fake_attempt(command, cwd=None):
        issued.append(list(command))
        return True

    monkeypatch.setattr(batch, "run", fake_run)
    monkeypatch.setattr(batch, "git_output", lambda *_: "targetvcpkg")
    monkeypatch.setattr(batch, "overlay_harness", lambda *_: issued.append(["<overlay>"]))
    monkeypatch.setattr(batch, "attempt", fake_attempt)

    batch.benchmark_commit(
        "abc",
        workspace=tmp_path,
        clone=tmp_path,
        harness_vcpkg="harnessvcpkg",
        skip_manifest=tmp_path / "skip.txt",
        run_id="1",
        api_url="https://benchview.test",
    )
    return issued


def position(issued: list[list[str]], needle: str) -> int:
    return next(i for i, command in enumerate(issued) if needle in " ".join(command))


def test_each_commit_starts_from_a_pristine_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The overlay leaves modified and untracked files, so checkout alone is not enough."""

    issued = record_commands(monkeypatch, tmp_path)
    assert position(issued, "reset --hard") < position(issued, "clean -xfd")
    assert position(issued, "clean -xfd") < position(issued, "checkout --detach abc")


def test_the_per_commit_stages_run_in_the_only_workable_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    issued = record_commands(monkeypatch, tmp_path)
    # ci.py setup resets every submodule, so the vcpkg repin has to follow it and
    # still precede configure, or the historical vcpkg comes back.
    assert position(issued, "ci.py setup") < position(issued, "checkout --detach harnessvcpkg")
    assert position(issued, "checkout --detach harnessvcpkg") < position(issued, "configure")
    # The overlay would be wiped by the build's own checkout dance, and the manifest
    # asks which of the overlaid benchmarks exist in the target.
    assert position(issued, "ci.py build") < position(issued, "<overlay>")
    assert position(issued, "<overlay>") < position(issued, "backfill_benchmark_manifest.py")
    assert position(issued, "backfill_benchmark_manifest.py") < position(issued, "benchmark-python")


def test_historical_sources_are_not_held_to_todays_warnings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """--cmake-args belongs to ci.py's top-level parser, so it must precede the subcommand."""

    issued = record_commands(monkeypatch, tmp_path)
    configure = next(c for c in issued if "configure" in c)
    assert configure[-2:] == ["--cmake-args=-DSGL_WARNINGS_AS_ERRORS=OFF", "configure"]


def test_the_torch_bridge_is_removed_even_when_the_benchmark_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The runners are persistent, so a stale bridge would reach the next commit."""

    issued: list[list[str]] = []

    def fake_run(command, cwd, stage, env=None):
        issued.append(list(command))
        if stage == "benchmark":
            raise batch.CommitFailure("benchmark", "exited with code 1")

    monkeypatch.setattr(batch, "run", fake_run)
    monkeypatch.setattr(batch, "git_output", lambda *_: "harnessvcpkg")
    monkeypatch.setattr(batch, "overlay_harness", lambda *_: None)
    monkeypatch.setattr(batch, "attempt", lambda command, cwd=None: issued.append(list(command)))

    with pytest.raises(batch.CommitFailure):
        batch.benchmark_commit(
            "abc",
            workspace=tmp_path,
            clone=tmp_path,
            harness_vcpkg="harnessvcpkg",
            skip_manifest=tmp_path / "skip.txt",
            run_id="1",
            api_url="https://benchview.test",
        )

    assert any("uninstall slangpy-torch" in " ".join(c) for c in issued)


def repin_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, is_ancestor: bool):
    """Run repin_vcpkg with a stubbed git, returning the commands it issued."""

    issued: list[list[str]] = []

    def fake_attempt(command, cwd=None):
        issued.append(list(command))
        return is_ancestor if "merge-base" in command else True

    monkeypatch.setattr(batch, "git_output", lambda *_: "targetvcpkg")
    monkeypatch.setattr(batch, "attempt", fake_attempt)
    monkeypatch.setattr(
        batch, "run", lambda command, cwd, stage, env=None: issued.append(list(command))
    )
    batch.repin_vcpkg(tmp_path, "harnessvcpkg")
    return issued


def test_an_unbootstrappable_vcpkg_pin_is_replaced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    issued = repin_calls(monkeypatch, tmp_path, is_ancestor=True)
    assert any("checkout --detach harnessvcpkg" in " ".join(c) for c in issued)


def test_a_merely_different_vcpkg_pin_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The dependency set is part of what is measured, so do not move it silently."""

    issued = repin_calls(monkeypatch, tmp_path, is_ancestor=False)
    assert not any("checkout" in " ".join(c) for c in issued)


def test_the_overlay_replaces_stale_harness_directories(tmp_path: Path):
    """A benchmark deleted from the harness must not survive in the clone."""

    workspace = tmp_path / "workspace"
    clone = tmp_path / "clone"
    for root in (workspace, clone):
        for relative in batch.OVERLAY_PATHS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith(".py"):
                path.write_text(f"# {root.name}\n", encoding="utf-8")
            else:
                path.mkdir(exist_ok=True)
                (path / "test_kept.py").write_text(f"# {root.name}\n", encoding="utf-8")
    (clone / "slangpy/benchmarks/test_retired.py").write_text("# stale\n", encoding="utf-8")

    batch.overlay_harness(workspace, clone)

    assert (clone / "tools/ci.py").read_text(encoding="utf-8") == "# workspace\n"
    assert not (clone / "slangpy/benchmarks/test_retired.py").exists()


def test_an_incomplete_harness_is_reported_rather_than_silently_skipped(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / "tools").mkdir(parents=True)
    with pytest.raises(batch.CommitFailure):
        batch.overlay_harness(workspace, tmp_path / "clone")


def test_history_is_cut_at_the_compatibility_floor():
    """Commits older than the floor cannot be built by the current harness."""

    history = commits(5)
    kept = dispatcher.supported_commits(history, history[2].sha)
    assert [c.sha for c in kept] == [c.sha for c in history[2:]]


def test_history_without_the_floor_is_rejected_rather_than_silently_truncated():
    with pytest.raises(dispatcher.UnsupportedHistoryError, match="deadbeef"):
        dispatcher.supported_commits(commits(3), "deadbeef")


def test_batches_cover_the_history_contiguously_and_exactly_once():
    history = commits(65)
    batches = dispatcher.batch_commits(history, 30)
    assert [len(b) for b in batches] == [30, 30, 5]
    assert [commit.sha for group in batches for commit in group] == [c.sha for c in history]


def test_a_batch_label_identifies_its_place_and_span():
    history = commits(65)
    batches = dispatcher.batch_commits(history, 30)
    assert dispatcher.batch_label(batches[2], 3, len(batches)) == (
        "batch 3/3 (5 commits, 2025-11-01..2025-11-05)"
    )


def git(repo: Path, *arguments: str) -> str:
    """Run git in ``repo`` with a fixed identity and return its stdout."""

    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(repo),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )
    return completed.stdout.strip()


def superproject_pinning(tmp_path: Path, *, old_first: bool) -> tuple[Path, str, str]:
    """Build a real superproject whose ``external/vcpkg`` gitlink points at one commit.

    Returns the superproject, the older vcpkg commit and the newer one. The submodule
    working tree is checked out at whichever of the two the superproject pins, which is
    the state ``ci.py setup`` leaves behind.
    """

    upstream = tmp_path / "vcpkg-origin"
    upstream.mkdir()
    git(upstream, "init", "-q", "-b", "main")
    (upstream / "ports.txt").write_text("old\n", encoding="utf-8")
    git(upstream, "add", "ports.txt")
    git(upstream, "commit", "-qm", "old vcpkg")
    older = git(upstream, "rev-parse", "HEAD")
    (upstream / "ports.txt").write_text("new\n", encoding="utf-8")
    git(upstream, "commit", "-qam", "new vcpkg")
    newer = git(upstream, "rev-parse", "HEAD")

    clone = tmp_path / "clone"
    clone.mkdir()
    git(clone, "init", "-q", "-b", "main")
    pinned = older if old_first else newer
    # A gitlink written directly, which is all repin_vcpkg reads, and far less
    # machinery than a real `git submodule add`.
    git(clone, "update-index", "--add", "--cacheinfo", f"160000,{pinned},external/vcpkg")
    git(clone, "commit", "-qm", "pin vcpkg")

    git(clone, "clone", "-q", str(upstream), str(clone / "external" / "vcpkg"))
    git(clone / "external" / "vcpkg", "checkout", "-q", "--detach", pinned)
    return clone, older, newer


def test_a_vcpkg_pin_older_than_the_harness_is_moved_forward(tmp_path: Path):
    """Prove an unbootstrappable vcpkg pin is actually replaced.

    The ancestry test has to be resolved against a revision that exists in the vcpkg
    repository. An earlier version compared against a slangpy commit, so the test could
    never succeed and every old pin was silently kept, which failed the build.
    """

    clone, older, newer = superproject_pinning(tmp_path, old_first=True)

    batch.repin_vcpkg(clone, newer)

    assert git(clone / "external" / "vcpkg", "rev-parse", "HEAD") == newer


def test_a_vcpkg_pin_the_harness_does_not_precede_is_left_alone(tmp_path: Path):
    """Keep a pin that is not older: the dependency set is part of the measurement."""

    clone, older, newer = superproject_pinning(tmp_path, old_first=False)

    batch.repin_vcpkg(clone, older)

    assert git(clone / "external" / "vcpkg", "rev-parse", "HEAD") == newer


def test_excluded_commits_are_dropped_before_batching(tmp_path: Path):
    history = commits(4)
    exclusions = tmp_path / "done.txt"
    exclusions.write_text(f"# already benchmarked\n{history[1].sha}\n\n", encoding="utf-8")

    remaining = [c for c in history if c.sha not in dispatcher.read_exclusions(exclusions)]

    assert [c.sha for c in remaining] == [history[0].sha, history[2].sha, history[3].sha]
