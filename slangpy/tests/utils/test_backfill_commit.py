# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Offline coverage for benchmarking one historical commit inside one backfill job."""

import os
from pathlib import Path
import subprocess
from typing import Optional, Sequence

import pytest

from tools import backfill_commit as driver


def record_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
    """Run one commit with every subprocess stubbed, returning the commands issued."""

    issued: list[list[str]] = []

    def fake_run(
        command: Sequence[str],
        cwd: Path,
        stage: str,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        issued.append(list(command))

    def fake_attempt(command: Sequence[str], cwd: Optional[Path] = None) -> bool:
        issued.append(list(command))
        return True

    monkeypatch.setattr(driver, "run", fake_run)
    monkeypatch.setattr(driver, "git_output", lambda *_: "targetvcpkg")
    monkeypatch.setattr(driver, "overlay_harness", lambda *_: issued.append(["<overlay>"]))
    monkeypatch.setattr(driver, "attempt", fake_attempt)
    # This is about the order of the per-commit stages, not about vcpkg ancestry,
    # and there is no repository here to resolve revisions against.
    monkeypatch.setattr(driver, "is_ancestor", lambda *_: True)

    driver.benchmark_commit(
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


def test_the_per_commit_stages_run_in_the_only_workable_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every one of these orderings produces silent garbage if it is inverted."""

    issued = record_commands(monkeypatch, tmp_path)

    # The overlay leaves modified and untracked files, so checkout alone is not enough.
    assert position(issued, "reset --hard") < position(issued, "clean -xfd")
    assert position(issued, "clean -xfd") < position(issued, "checkout --detach abc")
    # ci.py setup resets every submodule, so the vcpkg repin has to follow it and
    # still precede configure, or the historical vcpkg comes back.
    assert position(issued, "ci.py setup") < position(issued, "checkout --detach harnessvcpkg")
    assert position(issued, "checkout --detach harnessvcpkg") < position(issued, "configure")
    # The overlay would be wiped by the build's own checkout dance, and the manifest
    # asks which of the overlaid benchmarks exist in the target.
    assert position(issued, "ci.py build") < position(issued, "<overlay>")
    assert position(issued, "<overlay>") < position(issued, "backfill_benchmark_manifest.py")
    assert position(issued, "backfill_benchmark_manifest.py") < position(issued, "benchmark-python")
    # --cmake-args belongs to ci.py's top-level parser, so it must precede the subcommand.
    configure = next(c for c in issued if "configure" in c)
    assert configure[-2:] == ["--cmake-args=-DSGL_WARNINGS_AS_ERRORS=OFF", "configure"]


def test_the_torch_bridge_is_removed_even_when_the_benchmark_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runners are persistent, so a stale bridge would reach the next commit."""

    issued: list[list[str]] = []

    def fake_run(
        command: Sequence[str],
        cwd: Path,
        stage: str,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        issued.append(list(command))
        if stage == "benchmark":
            raise driver.CommitFailure("benchmark", "exited with code 1")

    monkeypatch.setattr(driver, "run", fake_run)
    monkeypatch.setattr(driver, "git_output", lambda *_: "harnessvcpkg")
    monkeypatch.setattr(driver, "overlay_harness", lambda *_: None)
    monkeypatch.setattr(driver, "attempt", lambda command, cwd=None: issued.append(list(command)))

    with pytest.raises(driver.CommitFailure):
        driver.benchmark_commit(
            "abc",
            workspace=tmp_path,
            clone=tmp_path,
            harness_vcpkg="harnessvcpkg",
            skip_manifest=tmp_path / "skip.txt",
            run_id="1",
            api_url="https://benchview.test",
        )

    assert any("uninstall slangpy-torch" in " ".join(c) for c in issued)


def test_the_overlay_replaces_the_harness_and_rejects_an_incomplete_one(tmp_path: Path) -> None:
    """A benchmark deleted from the harness must not survive in the clone."""

    workspace = tmp_path / "workspace"
    clone = tmp_path / "clone"
    for root in (workspace, clone):
        for relative in driver.OVERLAY_PATHS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith(".py"):
                path.write_text(f"# {root.name}\n", encoding="utf-8")
            else:
                path.mkdir(exist_ok=True)
                (path / "test_kept.py").write_text(f"# {root.name}\n", encoding="utf-8")
    (clone / "slangpy/benchmarks/test_retired.py").write_text("# stale\n", encoding="utf-8")

    driver.overlay_harness(workspace, clone)

    assert (clone / "tools/ci.py").read_text(encoding="utf-8") == "# workspace\n"
    assert not (clone / "slangpy/benchmarks/test_retired.py").exists()

    partial = tmp_path / "partial"
    (partial / "tools").mkdir(parents=True)
    with pytest.raises(driver.CommitFailure):
        driver.overlay_harness(partial, tmp_path / "other")


def git(repo: Path, *arguments: str) -> str:
    """Run git in ``repo`` with a fixed identity and return its stdout."""

    # A private HOME and empty config paths keep the developer's git configuration
    # out of the fixture repositories. os.devnull rather than /dev/null so these
    # tests run on the Windows CI leg too.
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
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
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


@pytest.mark.parametrize("target_is_older", [True, False])
def test_only_a_vcpkg_pin_older_than_the_harness_is_moved(
    tmp_path: Path, target_is_older: bool
) -> None:
    """Older pins cannot bootstrap and move; newer ones are part of the measurement.

    Resolved against a real repository rather than a stub, because ancestry can only
    be decided for revisions the vcpkg repository actually holds.
    """

    clone, older, newer = superproject_pinning(tmp_path, old_first=target_is_older)

    driver.repin_vcpkg(clone, newer if target_is_older else older)

    assert git(clone / "external" / "vcpkg", "rev-parse", "HEAD") == newer


def test_an_undecidable_vcpkg_comparison_is_reported_not_swallowed(tmp_path: Path) -> None:
    """A revision that does not resolve fails the commit rather than passing.

    ``git merge-base --is-ancestor`` exits 1 for "no" and 128 for "cannot tell", and
    reading the second as the first would silently disable the repin.
    """

    clone, older, _ = superproject_pinning(tmp_path, old_first=True)
    # A real commit, but from an unrelated repository, so the vcpkg clone cannot
    # resolve it and the fetch cannot supply it either.
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    git(stranger, "init", "-q", "-b", "main")
    (stranger / "unrelated.txt").write_text("x\n", encoding="utf-8")
    git(stranger, "add", "unrelated.txt")
    git(stranger, "commit", "-qm", "unrelated")
    unknown = git(stranger, "rev-parse", "HEAD")

    with pytest.raises(driver.CommitFailure, match="could not decide"):
        driver.repin_vcpkg(clone, unknown)

    # The pin is untouched, and the caller hears about it rather than inferring it.
    assert git(clone / "external" / "vcpkg", "rev-parse", "HEAD") == older
