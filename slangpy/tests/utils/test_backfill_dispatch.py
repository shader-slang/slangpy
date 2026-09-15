# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Coverage for dispatching one backfill run per commit and resuming a sweep.

The property under test throughout is that stopping the dispatcher and starting
it again is indistinguishable from never having stopped it: no commit is
benchmarked twice, and no dispatched run is forgotten.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Optional

import pytest

from tools import backfill_benchmarks as dispatcher
from tools import benchmark_actions as actions


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


def run(run_id: int, sha: str, status: str, conclusion: Optional[str] = None):
    """Build a workflow run titled the way the backfill workflow titles its runs."""

    return actions.WorkflowRun(
        run_id=run_id,
        display_title=f"backfill-benchmark: {sha}",
        status=status,
        conclusion=conclusion,
    )


class FakeGitHub:
    """Stand in for the GitHub CLI, recording dispatches and serving run states."""

    def __init__(self, runs=None, fail_dispatch_on=()):
        super().__init__()
        self.dispatched: list[str] = []
        self.runs: dict[int, actions.WorkflowRun] = dict(runs or {})
        self.listed: list[actions.WorkflowRun] = []
        self.fail_dispatch_on = set(fail_dispatch_on)
        self._next_id = 1000

    def dispatch_workflow(self, repository, workflow, workflow_ref, inputs):
        sha = inputs["target_sha"]
        if sha in self.fail_dispatch_on:
            raise actions.GitHubCliError("dispatch rejected")
        self.dispatched.append(sha)
        self._next_id += 1
        self.runs[self._next_id] = run(self._next_id, sha, "queued")
        return actions.DispatchResult(
            run_id=self._next_id,
            run_url=f"https://api.test/runs/{self._next_id}",
            html_url=f"https://github.test/runs/{self._next_id}",
        )

    def get_run(self, repository, run_id):
        return self.runs[run_id]

    def list_workflow_runs(self, repository, workflow, pages=5):
        return list(self.listed)

    def finish(self, run_id: int, conclusion: str) -> None:
        """Mark a dispatched run as completed with the given conclusion."""

        existing = self.runs[run_id]
        self.runs[run_id] = actions.WorkflowRun(
            run_id=run_id,
            display_title=existing.display_title,
            status="completed",
            conclusion=conclusion,
        )


def sweep(state, github, shas, max_in_flight=2, finish_as="success"):
    """Run a sweep to completion, completing each run as soon as it is polled."""

    def sleeper(_seconds):
        for run_id, workflow_run in list(github.runs.items()):
            if workflow_run.status != "completed":
                github.finish(run_id, finish_as)

    return dispatcher.sweep(
        state,
        github,
        shas,
        repository="owner/repo",
        workflow="backfill-benchmark.yml",
        workflow_ref="main",
        max_in_flight=max_in_flight,
        poll_seconds=0,
        sleeper=sleeper,
    )


def test_the_state_file_is_required_because_it_is_what_makes_a_sweep_resumable():
    """A sweep without a state file would silently re-benchmark everything."""

    with pytest.raises(SystemExit):
        dispatcher._parser().parse_args(["--max-in-flight", "2"])


def test_every_commit_is_dispatched_exactly_once(tmp_path: Path):
    state = dispatcher.SweepState.load(tmp_path / "state.json")
    github = FakeGitHub()
    history = [c.sha for c in commits(5)]

    assert sweep(state, github, history) == 0

    assert github.dispatched == history
    assert state.counts()[dispatcher.SUCCEEDED] == 5


def test_no_more_than_the_cap_is_ever_in_flight(tmp_path: Path):
    """The whole point of the cap is that the queue never runs away."""

    state = dispatcher.SweepState.load(tmp_path / "state.json")
    github = FakeGitHub()
    peak = 0

    def watching_sleeper(_seconds):
        nonlocal peak
        peak = max(peak, state.counts()[dispatcher.IN_FLIGHT])
        for run_id, workflow_run in list(github.runs.items()):
            if workflow_run.status != "completed":
                github.finish(run_id, "success")

    dispatcher.sweep(
        state,
        github,
        [c.sha for c in commits(9)],
        repository="owner/repo",
        workflow="backfill-benchmark.yml",
        workflow_ref="main",
        max_in_flight=3,
        poll_seconds=0,
        sleeper=watching_sleeper,
    )

    assert peak <= 3


def test_a_resumed_sweep_does_not_repeat_finished_commits(tmp_path: Path):
    """The headline property: restarting is as if the interruption never happened."""

    path = tmp_path / "state.json"
    history = [c.sha for c in commits(4)]

    first = dispatcher.SweepState.load(path)
    first.record(history[0], dispatcher.SUCCEEDED, 1)
    first.record(history[1], dispatcher.FAILED, 2)

    resumed = dispatcher.SweepState.load(path)
    github = FakeGitHub()
    pending = [sha for sha in history if resumed.status(sha) not in dispatcher.TERMINAL]

    assert sweep(resumed, github, pending) == 1
    # Only the two that had not reached an outcome are dispatched again.
    assert github.dispatched == history[2:]


def test_a_run_that_finished_while_we_were_away_is_adopted(tmp_path: Path):
    """An in-flight run is asked about, not dispatched a second time."""

    path = tmp_path / "state.json"
    sha = commits(1)[0].sha
    state = dispatcher.SweepState.load(path)
    state.record(sha, dispatcher.IN_FLIGHT, 4242)

    github = FakeGitHub(runs={4242: run(4242, sha, "completed", "success")})
    resumed = dispatcher.SweepState.load(path)
    dispatcher.reconcile(resumed, github, "owner/repo", "backfill-benchmark.yml")

    assert resumed.status(sha) == dispatcher.SUCCEEDED
    assert github.dispatched == []


def test_the_intent_to_dispatch_reaches_the_disk_before_the_run_exists(tmp_path: Path):
    """The precondition the recovery below depends on.

    If the marker were written after the API call, a crash in between would leave a
    run that no state file mentions, and the commit would later be dispatched again.
    """

    path = tmp_path / "state.json"
    state = dispatcher.SweepState.load(path)
    sha = commits(1)[0].sha
    observed: list[Optional[str]] = []

    class RecordingGitHub(FakeGitHub):
        def dispatch_workflow(self, repository, workflow, workflow_ref, inputs):
            # Read the file rather than the object: only what is on disk survives.
            payload = json.loads(path.read_text(encoding="utf-8"))
            observed.append(payload["commits"].get(inputs["target_sha"], {}).get("status"))
            return super().dispatch_workflow(repository, workflow, workflow_ref, inputs)

    dispatcher.dispatch(
        state,
        RecordingGitHub(),
        repository="owner/repo",
        workflow="backfill-benchmark.yml",
        workflow_ref="main",
        sha=sha,
    )

    assert observed == [dispatcher.DISPATCHING]


def test_a_dispatch_interrupted_before_its_run_was_recorded_is_recovered(tmp_path: Path):
    """The race the whole design turns on.

    If the dispatcher dies between the dispatch call and the state write, the run
    exists but its id was never recorded. Re-dispatching would benchmark the
    commit twice, so the run is found by the commit in its title instead.
    """

    path = tmp_path / "state.json"
    sha = commits(1)[0].sha
    state = dispatcher.SweepState.load(path)
    state.record(sha, dispatcher.DISPATCHING)

    github = FakeGitHub()
    github.listed = [run(777, sha, "in_progress")]
    resumed = dispatcher.SweepState.load(path)
    dispatcher.reconcile(resumed, github, "owner/repo", "backfill-benchmark.yml")

    assert resumed.status(sha) == dispatcher.IN_FLIGHT
    assert resumed.run_id(sha) == 777
    assert github.dispatched == []


def test_a_dispatch_that_never_reached_github_is_retried(tmp_path: Path):
    """The other side of the race: no run exists, so the commit is still pending."""

    path = tmp_path / "state.json"
    sha = commits(1)[0].sha
    state = dispatcher.SweepState.load(path)
    state.record(sha, dispatcher.DISPATCHING)

    github = FakeGitHub()
    github.listed = []
    resumed = dispatcher.SweepState.load(path)
    dispatcher.reconcile(resumed, github, "owner/repo", "backfill-benchmark.yml")

    assert resumed.status(sha) is None


def test_the_state_file_survives_an_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A half-written state file would strand every run it was tracking.

    The crash is injected partway through serialising, which is the only way to
    distinguish writing via a temporary file from writing the destination in place.
    """

    path = tmp_path / "state.json"
    state = dispatcher.SweepState.load(path)
    state.record("a" * 40, dispatcher.SUCCEEDED, 1)
    intact = path.read_text(encoding="utf-8")

    class Interrupted(Exception):
        """Stands in for the crash, without pytest's handling of KeyboardInterrupt."""

    def dump_then_die(payload, handle, **kwargs):
        handle.write('{"version": 1, "commits": {"bb')
        raise Interrupted

    # Mutated directly rather than through record(), which would save() and so
    # raise before the assertion below could be set up.
    state.commits["b" * 40] = {"status": dispatcher.IN_FLIGHT, "run_id": 2}
    monkeypatch.setattr(json, "dump", dump_then_die)
    with pytest.raises(Interrupted):
        state.save()
    monkeypatch.undo()

    # The destination never saw the partial write, so the earlier commit is still
    # readable and the sweep can be resumed.
    assert path.read_text(encoding="utf-8") == intact
    assert dispatcher.SweepState.load(path).status("a" * 40) == dispatcher.SUCCEEDED


def test_an_unreadable_state_file_stops_the_sweep(tmp_path: Path):
    """Overwriting it would re-dispatch every commit it was tracking."""

    path = tmp_path / "state.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(dispatcher.StateFileError):
        dispatcher.SweepState.load(path)


def test_the_state_file_records_progress_as_it_goes(tmp_path: Path):
    """Progress has to be on disk before the process ends, not written at exit."""

    path = tmp_path / "state.json"
    state = dispatcher.SweepState.load(path)
    github = FakeGitHub()
    history = [c.sha for c in commits(3)]

    observed: list[int] = []

    def sleeper(_seconds):
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed.append(len(payload["commits"]))
        for run_id, workflow_run in list(github.runs.items()):
            if workflow_run.status != "completed":
                github.finish(run_id, "success")

    dispatcher.sweep(
        state,
        github,
        history,
        repository="owner/repo",
        workflow="backfill-benchmark.yml",
        workflow_ref="main",
        max_in_flight=1,
        poll_seconds=0,
        sleeper=sleeper,
    )

    assert observed and observed[0] >= 1


def test_a_rejected_dispatch_keeps_its_place_in_the_queue(tmp_path: Path):
    """A commit GitHub refused must not be skipped over and silently lost."""

    state = dispatcher.SweepState.load(tmp_path / "state.json")
    history = [c.sha for c in commits(3)]
    github = FakeGitHub(fail_dispatch_on={history[0]})

    calls = {"n": 0}

    def sleeper(_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            # Whatever made the dispatch fail has cleared by the next attempt.
            github.fail_dispatch_on.clear()
        for run_id, workflow_run in list(github.runs.items()):
            if workflow_run.status != "completed":
                github.finish(run_id, "success")

    dispatcher.sweep(
        state,
        github,
        history,
        repository="owner/repo",
        workflow="backfill-benchmark.yml",
        workflow_ref="main",
        max_in_flight=2,
        poll_seconds=0,
        sleeper=sleeper,
    )

    assert github.dispatched == history
    assert state.status(history[0]) == dispatcher.SUCCEEDED


def test_a_failed_run_is_recorded_as_failed(tmp_path: Path):
    state = dispatcher.SweepState.load(tmp_path / "state.json")
    github = FakeGitHub()
    history = [c.sha for c in commits(2)]

    assert sweep(state, github, history, finish_as="failure") == 1
    assert state.counts()[dispatcher.FAILED] == 2


def test_supported_commits_start_at_the_floor():
    history = commits(5)
    selected = dispatcher.supported_commits(history, history[2].sha)
    assert [c.sha for c in selected] == [c.sha for c in history[2:]]


def test_history_without_the_floor_is_refused():
    """Missing the floor means the discovered history is not what was assumed."""

    # The message names the floor it looked for, because the usual cause is a
    # branch that does not contain it rather than a bug in the query.
    with pytest.raises(dispatcher.UnsupportedHistoryError, match="deadbeef"):
        dispatcher.supported_commits(commits(3), "deadbeef")


def test_a_run_title_identifies_its_commit():
    sha = "a" * 40
    assert dispatcher.sha_of(run(1, sha, "queued")) == sha
    assert dispatcher.sha_of(actions.WorkflowRun(1, "something else", "queued", None)) is None
