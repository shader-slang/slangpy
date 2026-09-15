# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Offline coverage for the GitHub adapter and the backfill batch dispatcher."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
from typing import Optional, Sequence

import pytest
import yaml

from tools import benchmark_actions as actions
from tools import dispatch_backfill_batches as dispatcher

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


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


class FakeCommandRunner:
    """Return queued ``gh`` results while retaining exact arguments and stdin."""

    def __init__(self, responses: Sequence[subprocess.CompletedProcess[str]]) -> None:
        """Copy deterministic responses so each invocation consumes one result."""

        super().__init__()
        self.responses = list(responses)
        self.calls: list[tuple[list[str], Optional[str]]] = []

    def __call__(
        self, arguments: Sequence[str], input_text: Optional[str]
    ) -> subprocess.CompletedProcess[str]:
        """Capture one no-shell invocation and return its next queued response."""

        self.calls.append((list(arguments), input_text))
        if not self.responses:
            raise AssertionError("Unexpected GitHub CLI invocation.")
        return self.responses.pop(0)


def completed_process(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    """Create one captured subprocess result for a fake command runner."""

    return subprocess.CompletedProcess(
        args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_paginated_commit_pages_are_flattened_in_order() -> None:
    """``gh --slurp`` returns an array of pages, which has to become one commit list."""

    first_sha, second_sha = "1" * 40, "2" * 40
    response = [
        [
            {
                "sha": sha,
                "html_url": f"https://github.test/commit/{sha}",
                "commit": {"message": "m", "committer": {"date": "2026-07-16T10:00:00Z"}},
            }
        ]
        for sha in (first_sha, second_sha)
    ]
    runner = FakeCommandRunner([completed_process(json.dumps(response))])
    github = actions.GitHubCli(command_runner=runner, executable="gh-test")

    commit_list = github.list_commits(
        "shader-slang/slangpy",
        "main",
        datetime(2026, 7, 16, tzinfo=timezone.utc),
        datetime(2026, 7, 17, tzinfo=timezone.utc),
    )

    assert [c.sha for c in commit_list] == [first_sha, second_sha]
    command, _ = runner.calls[0]
    assert "--paginate" in command and "--slurp" in command and "sha=main" in command


def test_a_dispatch_sends_its_body_on_stdin() -> None:
    """Inputs travel as JSON on stdin, never interpolated into the command line."""

    dispatch = {
        "workflow_run_id": 43,
        "run_url": "https://api.github.test/runs/43",
        "html_url": "https://github.test/runs/43",
    }
    runner = FakeCommandRunner([completed_process(json.dumps(dispatch))])
    github = actions.GitHubCli(command_runner=runner, executable="gh-test")

    result = github.dispatch_workflow(
        "shader-slang/slangpy",
        "ci-benchmark.yml",
        workflow_ref="main",
        inputs={"revision": "a" * 40},
    )

    assert result.run_id == 43
    _, input_text = runner.calls[0]
    assert json.loads(input_text or "") == {
        "ref": "main",
        "inputs": {"revision": "a" * 40},
        "return_run_details": True,
    }


def test_history_without_the_floor_is_rejected_rather_than_silently_truncated() -> None:
    """Silently dropping the floor would benchmark commits the harness cannot build."""

    history = commits(5)
    assert [c.sha for c in dispatcher.supported_commits(history, history[2].sha)] == [
        c.sha for c in history[2:]
    ]
    with pytest.raises(dispatcher.UnsupportedHistoryError, match="deadbeef"):
        dispatcher.supported_commits(commits(3), "deadbeef")


def test_batches_cover_the_history_contiguously_and_exactly_once() -> None:
    """A commit dropped or duplicated here is a silent hole in the series."""

    history = commits(65)
    batches = dispatcher.batch_commits(history, 30)
    assert [len(b) for b in batches] == [30, 30, 5]
    assert [c.sha for group in batches for c in group] == [c.sha for c in history]


def test_excluded_commits_are_dropped_before_batching(tmp_path: Path) -> None:
    """Blank lines and comments, indented or not, are not commits."""

    history = commits(4)
    exclusions = tmp_path / "done.txt"
    exclusions.write_text(f"  # already benchmarked\n{history[1].sha}\n\n", encoding="utf-8")

    excluded = dispatcher.read_exclusions(exclusions)

    assert [c.sha for c in history if c.sha not in excluded] == [
        history[0].sha,
        history[2].sha,
        history[3].sha,
    ]


def test_the_ordinary_workflow_benchmarks_only_a_validated_revision() -> None:
    """The job carries the BenchView write key and then runs the revision's own code.

    Checking out the raw input would let a branch or tag move between validation and
    checkout, and validating only the explicit input would leave the dispatch default
    able to benchmark an unmerged commit.
    """

    text = (REPOSITORY_ROOT / ".github/workflows/ci-benchmark.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    # YAML 1.1 reads an unquoted `on` as the boolean true, which is how GitHub spells
    # the trigger block.
    workflow["on"] = workflow.pop(True)
    resolved = "${{ needs.validate-revision.outputs.revision }}"

    build = workflow["jobs"]["build"]
    assert build["needs"] == "validate-revision"
    checkout = next(s for s in build["steps"] if "checkout" in s.get("uses", ""))
    assert checkout["with"]["ref"] == resolved
    assert build["env"]["BENCHVIEW_BENCHMARK_REF"] == resolved

    resolve = next(
        s for s in workflow["jobs"]["validate-revision"]["steps"] if s.get("id") == "resolve"
    )
    assert resolve["env"]["DEFAULT_REVISION"] == "${{ github.sha }}"
    assert 'requested="${REVISION:-$DEFAULT_REVISION}"' in resolve["run"]
    # One resolution and one ancestry check, so neither path can bypass the other.
    assert resolve["run"].count("git rev-parse") == 1
    assert resolve["run"].count("merge-base --is-ancestor") == 1
