# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Regression coverage for two ways a sweep can benchmark a commit twice.

Both shipped, and both were invisible to tests of the individual functions: the
dispatcher only double-dispatches when ``sweep`` is driven with a state that
``reconcile`` has already populated, which is exactly the combination a unit test
of either one alone never produces.
"""

from pathlib import Path

import pytest

from tools import backfill_benchmarks as dispatcher
from tools import benchmark_actions as actions


class FakeGitHub:
    """Record dispatches and let every run finish on the next poll."""

    def __init__(self) -> None:
        super().__init__()
        self.dispatched: list[str] = []
        self.runs: dict[int, actions.WorkflowRun] = {}
        self._next_id = 9000

    def dispatch_workflow(self, repository, workflow, workflow_ref, inputs):
        sha = inputs["target_sha"]
        self.dispatched.append(sha)
        self._next_id += 1
        self.runs[self._next_id] = actions.WorkflowRun(
            self._next_id, f"backfill-benchmark: {sha}", "queued", None
        )
        return actions.DispatchResult(self._next_id, "run-url", "html-url")

    def get_run(self, repository, run_id):
        return self.runs[run_id]

    def list_workflow_runs(self, repository, workflow, pages=5):
        return []

    def finish_everything(self, _seconds: float) -> None:
        for run_id, run in list(self.runs.items()):
            if run.status != "completed":
                self.runs[run_id] = actions.WorkflowRun(
                    run_id, run.display_title, "completed", "success"
                )


def test_a_run_reconcile_adopted_is_not_dispatched_a_second_time(tmp_path: Path):
    """The commit is already running; a second run would measure it twice.

    The state file keeps only the newer run id, so the first run also becomes
    unreachable: nothing would ever collect its result.
    """

    adopted, pending = "a" * 40, "b" * 40
    state = dispatcher.SweepState.load(tmp_path / "state.json")
    state.record(adopted, dispatcher.IN_FLIGHT, 4242)

    github = FakeGitHub()
    github.runs[4242] = actions.WorkflowRun(
        4242, f"backfill-benchmark: {adopted}", "in_progress", None
    )

    dispatcher.sweep(
        state,
        github,
        [adopted, pending],
        repository="owner/repo",
        workflow="backfill-benchmark.yml",
        workflow_ref="main",
        # More than one, so the cap does not hide the bug by leaving no headroom.
        max_in_flight=2,
        poll_seconds=0,
        sleeper=github.finish_everything,
    )

    assert github.dispatched == [pending]
    assert state.run_id(adopted) == 4242


@pytest.mark.parametrize(
    "commits",
    ['{"aa": "not an object"}', '{"aa": {"status": "invented"}}'],
)
def test_a_record_that_cannot_be_read_stops_the_sweep(tmp_path: Path, commits: str):
    """Discarding it would make the commit look untouched and dispatch it again."""

    path = tmp_path / "state.json"
    path.write_text(f'{{"version": {dispatcher.STATE_VERSION}, "commits": {commits}}}')

    with pytest.raises(dispatcher.StateFileError):
        dispatcher.SweepState.load(path)


def test_a_boolean_is_not_accepted_as_a_run_id(tmp_path: Path):
    """bool is an int subclass, so True would be formatted into a run endpoint."""

    state = dispatcher.SweepState.load(tmp_path / "state.json")
    state.commits["a" * 40] = {"status": dispatcher.IN_FLIGHT, "run_id": True}

    assert state.run_id("a" * 40) is None
