# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Offline coverage for the GitHub adapter and the benchmark workflow definitions."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Optional, Sequence

import pytest

from tools import benchmark_actions as actions
from tools import dispatch_backfill_batches as dispatcher

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


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
        args=["gh"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def make_run_document(index: int) -> dict[str, Any]:
    """Create one GitHub workflow-run response object for adapter pagination tests."""

    sha = f"{index:040x}"
    return {
        "id": index,
        "display_title": f"backfill-benchmark: {sha}",
        "status": "completed",
        "conclusion": "success",
        "html_url": f"https://github.test/runs/{index}",
        "head_sha": sha,
        "created_at": "2026-07-16T10:00:00Z",
        "updated_at": "2026-07-16T11:00:00Z",
    }


def test_github_cli_parses_paginated_commits_and_uses_versioned_api() -> None:
    """Flatten commit pages while passing all filters through an argument-array command."""

    first_sha = "1" * 40
    second_sha = "2" * 40
    response = [
        [
            {
                "sha": first_sha,
                "html_url": f"https://github.test/commit/{first_sha}",
                "commit": {
                    "message": "First",
                    "committer": {"date": "2026-07-16T10:00:00Z"},
                },
            }
        ],
        [
            {
                "sha": second_sha,
                "html_url": f"https://github.test/commit/{second_sha}",
                "commit": {
                    "message": "Second",
                    "committer": {"date": "2026-07-16T11:00:00Z"},
                },
            }
        ],
    ]
    runner = FakeCommandRunner([completed_process(json.dumps(response))])
    github = actions.GitHubCli(command_runner=runner, executable="gh-test")

    commits = github.list_commits(
        "shader-slang/slangpy",
        "main",
        datetime(2026, 7, 16, tzinfo=timezone.utc),
        datetime(2026, 7, 17, tzinfo=timezone.utc),
    )

    assert [commit.sha for commit in commits] == [first_sha, second_sha]
    command, input_text = runner.calls[0]
    assert command[:2] == ["gh-test", "api"]
    assert f"X-GitHub-Api-Version: {actions.GITHUB_API_VERSION}" in command
    assert "--paginate" in command
    assert "--slurp" in command
    assert "sha=main" in command
    assert input_text is None


def test_github_cli_parses_runs_and_dispatches_json_on_stdin() -> None:
    """Parse workflow pages and return immediate run details from exact dispatch JSON."""

    run_page = {
        "workflow_runs": [
            {
                "id": 42,
                "display_title": "ci-benchmark: " + "a" * 40,
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.test/runs/42",
                "head_sha": "b" * 40,
                "created_at": "2026-07-16T10:00:00Z",
                "updated_at": "2026-07-16T11:00:00Z",
            }
        ]
    }
    dispatch = {
        "workflow_run_id": 43,
        "run_url": "https://api.github.test/runs/43",
        "html_url": "https://github.test/runs/43",
    }
    runner = FakeCommandRunner(
        [completed_process(json.dumps(run_page)), completed_process(json.dumps(dispatch))]
    )
    github = actions.GitHubCli(command_runner=runner, executable="gh-test")

    runs = github.list_workflow_runs("shader-slang/slangpy", "ci-benchmark.yml", maximum=10)
    result = github.dispatch_workflow(
        "shader-slang/slangpy",
        "ci-benchmark.yml",
        workflow_ref="main",
        inputs={"revision": "a" * 40},
    )

    assert runs[0].run_id == 42
    assert runs[0].title == "ci-benchmark: " + "a" * 40
    assert result.run_id == 43
    command, input_text = runner.calls[1]
    assert "--method" in command and "POST" in command
    assert "--input" in command and "-" in command
    assert json.loads(input_text or "") == {
        "ref": "main",
        "inputs": {"revision": "a" * 40},
        "return_run_details": True,
    }


def test_github_cli_fetches_only_the_requested_workflow_run_pages() -> None:
    """Bound capacity polling instead of downloading the workflow's complete history."""

    first_page = {"workflow_runs": [make_run_document(index) for index in range(1, 101)]}
    second_page = {"workflow_runs": [make_run_document(101)]}
    runner = FakeCommandRunner(
        [completed_process(json.dumps(first_page)), completed_process(json.dumps(second_page))]
    )
    github = actions.GitHubCli(command_runner=runner, executable="gh-test")

    runs = github.list_workflow_runs("shader-slang/slangpy", "backfill-benchmark.yml", maximum=101)

    assert len(runs) == 101
    assert len(runner.calls) == 2
    first_command, _ = runner.calls[0]
    second_command, _ = runner.calls[1]
    assert "per_page=100" in first_command
    assert "page=1" in first_command
    assert "page=2" in second_command
    assert "--paginate" not in first_command
    assert f"X-GitHub-Api-Version: {actions.GITHUB_API_VERSION}" in first_command


def test_github_cli_reports_command_and_installation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give operators actionable errors without requiring a real binary or printing tokens."""

    runner = FakeCommandRunner(
        [completed_process("", returncode=1, stderr="authentication failed")]
    )
    github = actions.GitHubCli(command_runner=runner, executable="gh-test")
    with pytest.raises(actions.GitHubCliError, match="gh auth status"):
        github.list_workflow_runs("shader-slang/slangpy", "ci-benchmark.yml", maximum=1)

    monkeypatch.setattr(actions.shutil, "which", lambda executable: None)
    with pytest.raises(actions.GitHubCliError, match="gh auth login"):
        actions.GitHubCli()


def test_default_gh_runner_never_uses_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lock the real subprocess boundary to captured UTF-8 argument-array execution."""

    captured: dict[str, Any] = {}

    def fake_run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Capture subprocess options without starting an external program."""

        captured["arguments"] = arguments
        captured.update(kwargs)
        return completed_process("{}")

    monkeypatch.setattr(actions.subprocess, "run", fake_run)
    actions._default_command_runner(["gh", "api", "rate_limit"], None)

    assert captured["arguments"] == ["gh", "api", "rate_limit"]
    assert captured["shell"] is False
    assert captured["check"] is False
    assert captured["capture_output"] is True
    assert captured["encoding"] == "utf-8"


def test_ordinary_workflow_selects_tip_or_exact_revision_without_backfill_logic() -> None:
    """Keep exact future selection generic and historical overlays out of ordinary CI."""

    workflow = (REPOSITORY_ROOT / ".github/workflows/ci-benchmark.yml").read_text(encoding="utf-8")

    assert "cron:" not in workflow
    assert 'run-name: "ci-benchmark: ${{ inputs.revision || github.sha }}"' in workflow
    assert "revision:" in workflow
    assert "ref: ${{ inputs.revision || github.sha }}" in workflow
    assert "BENCHVIEW_BENCHMARK_REF: ${{ inputs.revision || github.sha }}" in workflow
    assert "BENCHVIEW_BENCHMARK_BRANCH: ${{ github.ref_name }}" in workflow
    assert "target_sha" not in workflow
    assert "Overlay current BenchView benchmark harness" not in workflow
    assert "run_benchmark_ci.py" not in workflow


def test_nightly_workflow_dispatches_recent_commits_with_github_script() -> None:
    """Keep nightly fan-out inside GitHub Actions without Python or GitHub CLI setup."""

    workflow = (REPOSITORY_ROOT / ".github/workflows/schedule-benchmarks.yml").read_text(
        encoding="utf-8"
    )

    assert "actions/github-script@v9" in workflow
    assert "github.rest.repos.listCommits" in workflow
    assert "github.rest.actions.createWorkflowDispatch" in workflow
    assert 'const workflow = "ci-benchmark.yml"' in workflow
    assert 'const branch = "main"' in workflow
    assert "inputs: { revision }" in workflow
    assert "commits.reverse()" in workflow
    assert "github.rest.actions.listWorkflowRuns" not in workflow
    assert "existingTitles" not in workflow
    assert "dry_run" not in workflow
    assert "python" not in workflow.lower()
    assert "gh --version" not in workflow
    assert "actions/checkout" not in workflow


def test_backfill_workflow_batches_a_rev_list_and_cleans_up_safely() -> None:
    """Lock the batched workflow's inputs, timeout, and clone deletion guard."""

    workflow = (REPOSITORY_ROOT / ".github/workflows/backfill-benchmark.yml").read_text(
        encoding="utf-8"
    )

    # The whole point of the batched design: one run covers a list of commits.
    assert "target_shas:" in workflow
    assert "target_sha:" not in workflow
    # 30 Windows commits is about five hours, too close to the six hour default.
    assert "timeout-minutes: 720" in workflow
    # The rev list reaches the driver through the environment so that the step is
    # written once and works under both pwsh and bash.
    assert "BACKFILL_TARGET_SHAS: ${{ inputs.target_shas }}" in workflow
    assert '"${{ inputs.target_shas }}"' not in workflow
    assert "python tools/backfill_batch.py" in workflow
    # The driver needs the harness checkout to copy from and the floor to validate against.
    assert "actions/checkout" in workflow
    assert dispatcher.SUPPORTED_FLOOR_SHA in workflow
    assert (
        "${{ runner.temp }}/slangpy-backfill-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.os }}"
        in workflow
    )
    assert 'git clone --recursive "https://github.com/${{ github.repository }}.git"' in workflow
    # BenchView's ref varies per commit now, so only the branch can be a job-level constant.
    assert "BENCHVIEW_BENCHMARK_BRANCH: main" in workflow
    assert "BENCHVIEW_BENCHMARK_REF:" not in workflow
    assert "[StringComparer]::OrdinalIgnoreCase.Equals($parent.FullName, $runnerTemp)" in workflow
    assert '$name.StartsWith("slangpy-backfill-"' in workflow
    assert "Remove-Item -LiteralPath $candidate -Recurse -Force" in workflow
    assert '"$(dirname -- "$candidate")" != "$runner_temp"' in workflow
    assert '"$(basename -- "$candidate")" != slangpy-backfill-*' in workflow
    assert 'rm -rf -- "$candidate"' in workflow
