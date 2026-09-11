# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Offline coverage for ordinary, nightly, and historical benchmark scheduling."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Optional, Sequence

import pytest

from tools import backfill_benchmarks as backfill
from tools import benchmark_actions as actions

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)


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


class FakeGitHub(actions.GitHubCli):
    """Provide in-memory commit, run, and dispatch behavior to scheduler tests."""

    def __init__(
        self,
        commits: Sequence[actions.Commit],
        runs: Sequence[actions.WorkflowRun] = (),
    ) -> None:
        """Seed fake GitHub history without locating or executing a real ``gh`` binary."""

        super().__init__(command_runner=FakeCommandRunner([]), executable="gh-test")
        self.commits = list(commits)
        self.runs = list(runs)
        self.dispatches: list[tuple[str, str, str, dict[str, str]]] = []
        self.next_run_id = 9000

    def list_commits(
        self,
        repository: str,
        branch: str,
        since: datetime,
        until: datetime,
    ) -> list[actions.Commit]:
        """Return seeded commits inside the scheduler's explicit UTC interval."""

        del repository, branch
        return [commit for commit in self.commits if since <= commit.committed_at <= until]

    def list_workflow_runs(
        self, repository: str, workflow: str, maximum: int
    ) -> list[actions.WorkflowRun]:
        """Return the newest requested number of seeded workflow runs."""

        del repository, workflow
        return self.runs[:maximum]

    def dispatch_workflow(
        self,
        repository: str,
        workflow: str,
        workflow_ref: str,
        inputs: dict[str, str],
    ) -> actions.DispatchResult:
        """Record one dispatch and expose it to subsequent title reconciliation."""

        self.dispatches.append((repository, workflow, workflow_ref, dict(inputs)))
        sha = inputs.get("revision") or inputs.get("target_sha")
        if sha is None:
            raise AssertionError("Benchmark workflow dispatch lacks its revision input.")
        title_prefix = "ci-benchmark" if "revision" in inputs else "backfill-benchmark"
        result = actions.DispatchResult(
            run_id=self.next_run_id,
            run_url=f"https://api.github.test/runs/{self.next_run_id}",
            html_url=f"https://github.test/runs/{self.next_run_id}",
        )
        self.runs.insert(
            0,
            make_run(
                run_id=result.run_id,
                title=f"{title_prefix}: {sha}",
                status="queued",
                sha=sha,
            ),
        )
        self.next_run_id += 1
        return result


class FailingDispatchGitHub(FakeGitHub):
    """Simulate an ambiguous transport failure after write-ahead state publication."""

    def dispatch_workflow(
        self,
        repository: str,
        workflow: str,
        workflow_ref: str,
        inputs: dict[str, str],
    ) -> actions.DispatchResult:
        """Fail without revealing whether GitHub accepted the request."""

        del repository, workflow, workflow_ref, inputs
        raise actions.GitHubCliError("connection lost")


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


def make_commit(index: int, committed_at: Optional[datetime] = None) -> actions.Commit:
    """Create one stable 40-character commit record for scheduler scenarios."""

    sha = f"{index:040x}"
    return actions.Commit(
        sha=sha,
        committed_at=committed_at or NOW + timedelta(minutes=index),
        message=f"Commit {index}\n\nDetails",
        html_url=f"https://github.test/commit/{sha}",
    )


def floor_commit() -> actions.Commit:
    """Create the exact inclusive compatibility-boundary commit."""

    return actions.Commit(
        sha=backfill.SUPPORTED_FLOOR_SHA,
        committed_at=backfill.SUPPORTED_FLOOR_TIME,
        message="Device-isolated benchmark support",
        html_url=f"https://github.test/commit/{backfill.SUPPORTED_FLOOR_SHA}",
    )


def make_run(
    run_id: int,
    title: str,
    status: str,
    sha: str = "f" * 40,
) -> actions.WorkflowRun:
    """Create one workflow run with stable reconciliation and capacity fields."""

    return actions.WorkflowRun(
        run_id=run_id,
        title=title,
        status=status,
        conclusion=None if status != "completed" else "success",
        html_url=f"https://github.test/runs/{run_id}",
        head_sha=sha,
        created_at=NOW + timedelta(seconds=run_id),
        updated_at=NOW + timedelta(seconds=run_id),
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


def make_store(path: Path) -> backfill.BackfillStateStore:
    """Create one test state store bound to the production backfill configuration."""

    return backfill.BackfillStateStore(
        path,
        repository=backfill.DEFAULT_REPOSITORY,
        branch=backfill.DEFAULT_BRANCH,
        workflow=backfill.DEFAULT_WORKFLOW,
        lower_bound=backfill.SUPPORTED_FLOOR_SHA,
    )


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


def test_backfill_workflow_guards_boundary_builds_before_overlay_and_cleans_safely() -> None:
    """Lock the historical workflow ordering, source override, and deletion guard."""

    workflow = (REPOSITORY_ROOT / ".github/workflows/backfill-benchmark.yml").read_text(
        encoding="utf-8"
    )

    assert 'run-name: "backfill-benchmark: ${{ inputs.target_sha }}"' in workflow
    assert "required: true" in workflow
    assert (
        "${{ runner.temp }}/slangpy-backfill-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.os }}"
        in workflow
    )
    assert 'git clone --recursive "https://github.com/${{ github.repository }}.git"' in workflow
    assert 'checkout --detach "${{ inputs.target_sha }}"' in workflow
    assert "git submodule sync --recursive" in workflow
    assert "git submodule update --init --recursive" in workflow
    assert "git lfs pull" in workflow
    assert "merge-base --is-ancestor" in workflow
    assert backfill.SUPPORTED_FLOOR_SHA in workflow
    assert workflow.index("Validate supported history boundary") < workflow.index(
        "Historical setup"
    )
    assert workflow.index("Historical build") < workflow.index(
        "Overlay current BenchView benchmark harness"
    )
    assert workflow.index("Overlay current BenchView benchmark harness") < workflow.index(
        "Benchmark historical source"
    )
    assert (
        'git checkout "${{ github.sha }}" -- tools/ci.py tools/gpu_clock.py '
        "slangpy/testing/benchmark"
    ) in workflow
    assert "python tools/ci.py benchmark-python" in workflow
    assert "BENCHVIEW_BENCHMARK_REF: ${{ inputs.target_sha }}" in workflow
    assert "BENCHVIEW_BENCHMARK_BRANCH: main" in workflow
    assert "shell: python" not in workflow
    assert "[StringComparer]::OrdinalIgnoreCase.Equals($parent.FullName, $runnerTemp)" in workflow
    assert '$name.StartsWith("slangpy-backfill-"' in workflow
    assert "Remove-Item -LiteralPath $candidate -Recurse -Force" in workflow
    assert '"$(dirname -- "$candidate")" != "$runner_temp"' in workflow
    assert '"$(basename -- "$candidate")" != slangpy-backfill-*' in workflow
    assert 'rm -rf -- "$candidate"' in workflow


def test_boundary_rejection_excludes_history_before_the_floor() -> None:
    """Reject an inventory that cannot prove the configured inclusive floor."""

    with pytest.raises(backfill.BackfillStateError, match=backfill.SUPPORTED_FLOOR_SHA):
        backfill.supported_commits([make_commit(1)], backfill.SUPPORTED_FLOOR_SHA)


def test_backfill_with_three_active_runs_dispatches_one_oldest_commit(
    tmp_path: Path,
) -> None:
    """Fill the fourth workflow slot with exactly one oldest pending revision."""

    commits = [floor_commit(), make_commit(2, backfill.SUPPORTED_FLOOR_TIME + timedelta(days=1))]
    active_runs = [
        make_run(index, f"unrelated backfill {index}", "in_progress") for index in range(3)
    ]
    github = FakeGitHub(commits, active_runs)
    store = make_store(tmp_path / "state.json")

    result = backfill.run_backfill_scheduler(
        github,
        store,
        now_provider=lambda: NOW,
        sleep=lambda seconds: None,
        poll_seconds=60,
        once=True,
        dry_run=False,
        output=lambda line: None,
    )

    assert result == 0
    assert len(github.dispatches) == 1
    assert github.dispatches[0][3] == {"target_sha": backfill.SUPPORTED_FLOOR_SHA}
    reloaded = store.load()
    floor = reloaded.records[backfill.SUPPORTED_FLOOR_SHA]
    assert floor.status == "dispatched"
    assert floor.run_id == 9000
    assert floor.run_url == "https://github.test/runs/9000"
    assert list(tmp_path.glob("state.json.*.tmp")) == []


def test_backfill_with_four_active_runs_dispatches_nothing(tmp_path: Path) -> None:
    """Hold all pending history when the four-workflow pressure limit is full."""

    commits = [floor_commit(), make_commit(2, backfill.SUPPORTED_FLOOR_TIME + timedelta(days=1))]
    active_runs = [make_run(index, f"unrelated backfill {index}", "queued") for index in range(4)]
    github = FakeGitHub(commits, active_runs)
    store = make_store(tmp_path / "state.json")

    backfill.run_backfill_scheduler(
        github,
        store,
        now_provider=lambda: NOW,
        sleep=lambda seconds: None,
        poll_seconds=60,
        once=True,
        dry_run=False,
        output=lambda line: None,
    )

    assert github.dispatches == []
    assert store.load().records[backfill.SUPPORTED_FLOOR_SHA].status == "pending"


def test_backfill_restart_preserves_the_one_minute_dispatch_interval(tmp_path: Path) -> None:
    """Prevent a quick restart from bypassing the persisted dispatch cadence."""

    second = make_commit(2, backfill.SUPPORTED_FLOOR_TIME + timedelta(days=1))
    commits = [floor_commit(), second]
    existing = make_run(
        70,
        backfill.backfill_run_title(backfill.SUPPORTED_FLOOR_SHA),
        "in_progress",
        backfill.SUPPORTED_FLOOR_SHA,
    )
    github = FakeGitHub(commits, [existing])
    store = make_store(tmp_path / "state.json")
    state = store.load()
    backfill.merge_discovered_commits(state, commits)
    floor = state.records[backfill.SUPPORTED_FLOOR_SHA]
    floor.status = "dispatched"
    floor.dispatch_started_at = NOW - timedelta(seconds=30)
    floor.run_id = existing.run_id
    floor.run_url = existing.html_url
    store.save(state)

    backfill.run_backfill_scheduler(
        github,
        store,
        now_provider=lambda: NOW,
        sleep=lambda seconds: None,
        poll_seconds=60,
        once=True,
        dry_run=False,
        output=lambda line: None,
    )
    assert github.dispatches == []

    backfill.run_backfill_scheduler(
        github,
        store,
        now_provider=lambda: NOW + timedelta(seconds=31),
        sleep=lambda seconds: None,
        poll_seconds=60,
        once=True,
        dry_run=False,
        output=lambda line: None,
    )
    assert [dispatch[3] for dispatch in github.dispatches] == [{"target_sha": second.sha}]


def test_incompatible_state_is_rejected_without_modification(tmp_path: Path) -> None:
    """Require explicit archival of a prototype state bound to another workflow."""

    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "repository": backfill.DEFAULT_REPOSITORY,
                "branch": backfill.DEFAULT_BRANCH,
                "workflow": "ci-benchmark.yml",
                "lowerBound": backfill.SUPPORTED_FLOOR_SHA,
                "commits": [],
            }
        ),
        encoding="utf-8",
    )
    original = path.read_bytes()

    with pytest.raises(backfill.BackfillStateError, match="Archive"):
        make_store(path).load()

    assert path.read_bytes() == original


def test_uncertain_dispatch_reconciles_by_title_or_expires_after_grace(tmp_path: Path) -> None:
    """Recover a crash after POST without duplicating an accepted workflow request."""

    store = make_store(tmp_path / "state.json")
    state = store.load()
    commit = floor_commit()
    backfill.merge_discovered_commits(state, [commit])
    record = state.records[commit.sha]
    record.status = "dispatching"
    record.dispatch_started_at = NOW
    store.save(state)

    accepted = make_run(77, backfill.backfill_run_title(commit.sha), "queued", commit.sha)
    assert backfill.reconcile_records(
        state, [accepted], NOW + timedelta(minutes=1), backfill.DEFAULT_DISPATCH_GRACE
    )
    assert record.status == "dispatched"
    assert record.run_id == 77

    record.status = "dispatching"
    record.dispatch_started_at = NOW
    record.run_id = None
    record.run_url = None
    assert not backfill.reconcile_records(
        state, [], NOW + timedelta(minutes=9), backfill.DEFAULT_DISPATCH_GRACE
    )
    assert record.status == "dispatching"
    assert backfill.reconcile_records(
        state, [], NOW + timedelta(minutes=10), backfill.DEFAULT_DISPATCH_GRACE
    )
    assert record.status == "pending"


def test_scheduler_waits_for_an_uncertain_only_record_then_retries(tmp_path: Path) -> None:
    """Keep running through the grace window when no immediately pending commit remains."""

    commit = floor_commit()
    github = FakeGitHub([commit])
    store = make_store(tmp_path / "state.json")
    state = store.load()
    backfill.merge_discovered_commits(state, [commit])
    record = state.records[commit.sha]
    record.status = "dispatching"
    record.dispatch_started_at = NOW
    store.save(state)
    times = iter(
        [
            NOW + timedelta(minutes=1),
            NOW + timedelta(minutes=1),
            NOW + timedelta(minutes=11),
        ]
    )
    sleeps: list[float] = []

    result = backfill.run_backfill_scheduler(
        github,
        store,
        now_provider=lambda: next(times),
        sleep=sleeps.append,
        poll_seconds=60,
        once=False,
        dry_run=False,
        output=lambda line: None,
    )

    assert result == 0
    assert sleeps == [60]
    assert [dispatch[3] for dispatch in github.dispatches] == [{"target_sha": commit.sha}]
    assert store.load().records[commit.sha].status == "dispatched"


def test_failed_dispatch_leaves_write_ahead_state_for_restart_reconciliation(
    tmp_path: Path,
) -> None:
    """Preserve an ambiguous request until a deterministic GitHub title resolves it."""

    store = make_store(tmp_path / "state.json")
    state = store.load()
    commit = floor_commit()
    backfill.merge_discovered_commits(state, [commit])
    github = FailingDispatchGitHub([commit])

    with pytest.raises(actions.GitHubCliError, match="connection lost"):
        backfill.dispatch_oldest_pending(github, store, state, NOW)

    reloaded = store.load()
    assert reloaded.records[commit.sha].status == "dispatching"
    accepted = make_run(88, backfill.backfill_run_title(commit.sha), "queued", commit.sha)
    assert backfill.reconcile_records(
        reloaded, [accepted], NOW + timedelta(minutes=1), backfill.DEFAULT_DISPATCH_GRACE
    )
    assert reloaded.records[commit.sha].status == "dispatched"
    assert reloaded.records[commit.sha].run_id == 88


def test_backfill_dry_run_does_not_create_state_or_dispatch(tmp_path: Path) -> None:
    """Preview the complete supported inventory without any local or GitHub write."""

    commits = [floor_commit(), make_commit(2, backfill.SUPPORTED_FLOOR_TIME + timedelta(days=1))]
    github = FakeGitHub(commits)
    path = tmp_path / "state.json"
    output: list[str] = []

    backfill.run_backfill_scheduler(
        github,
        make_store(path),
        now_provider=lambda: NOW,
        sleep=lambda seconds: None,
        poll_seconds=60,
        once=False,
        dry_run=True,
        output=output.append,
    )

    assert not path.exists()
    assert github.dispatches == []
    assert any(backfill.DEFAULT_WORKFLOW in line for line in output)
    assert any(backfill.SUPPORTED_FLOOR_SHA in line for line in output)


def test_backfill_main_returns_130_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate an operator Ctrl+C into the documented durable-restart exit status."""

    def make_github() -> FakeGitHub:
        """Avoid locating or contacting a real GitHub CLI during the command test."""

        return FakeGitHub([floor_commit()])

    def interrupt_scheduler(*args: Any, **kwargs: Any) -> int:
        """Interrupt after command setup as if the operator pressed Ctrl+C."""

        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(backfill, "GitHubCli", make_github)
    monkeypatch.setattr(backfill, "run_backfill_scheduler", interrupt_scheduler)

    result = backfill.main(["--state-file", str(tmp_path / "state.json")])

    assert result == 130
