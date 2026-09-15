# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dispatch one backfill run per commit, keeping a bounded number in flight.

The supported history is hundreds of commits and the performance runner pool is
a couple of machines, so the whole sweep cannot simply be submitted at once.
This process supervises it instead, holding a bounded number of runs in flight
and dispatching the next commit as an earlier one finishes.

That makes it long-running, and therefore something that will be interrupted.
The state file is mandatory for exactly that reason. It is rewritten atomically
after every state transition, so stopping this process and starting it again is
indistinguishable from never having stopped it: commits already benchmarked are
not repeated, and runs still in flight are adopted rather than dispatched a
second time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable, Iterable, Optional, Sequence

# Importable both as ``tools.backfill_benchmarks`` and as a directly executed
# script, which only puts tools/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.backfill_commit import SUPPORTED_FLOOR_SHA, SUPPORTED_FLOOR_TIME
from tools.benchmark_actions import Commit, GitHubCli, GitHubCliError, WorkflowRun

DEFAULT_REPOSITORY = "shader-slang/slangpy"
DEFAULT_WORKFLOW = "backfill-benchmark.yml"

# One run occupies both perf machines at once, because its Linux and Windows jobs
# are dispatched together. Two keeps a run queued behind the running one so a
# machine is not idle between commits, without building a queue so long that
# stopping the sweep means waiting for it to drain.
DEFAULT_MAX_IN_FLIGHT = 2
DEFAULT_POLL_SECONDS = 60

# Individual dispatches are retried, because rate limiting and transient API
# failures are expected over a sweep this long. A run of failures this size is
# not transient, and looping on it would hide the real problem behind progress
# output. The state file makes stopping cheap.
MAX_CONSECUTIVE_DISPATCH_FAILURES = 5

STATE_VERSION = 1

# A commit is in exactly one of these. Anything absent from the state file is
# pending, so a fresh file and a file listing no commits mean the same thing.
DISPATCHING = "dispatching"
IN_FLIGHT = "in_flight"
SUCCEEDED = "succeeded"
FAILED = "failed"
TERMINAL = (SUCCEEDED, FAILED)
KNOWN_STATUSES = (DISPATCHING, IN_FLIGHT, SUCCEEDED, FAILED)


class UnsupportedHistoryError(RuntimeError):
    """The discovered history does not contain the configured compatibility floor."""


class StateFileError(RuntimeError):
    """The state file exists but is not a state file this version can use."""


def supported_commits(commits: Sequence[Commit], lower_bound: str) -> list[Commit]:
    """Return commits from the inclusive boundary forward in chronological order.

    :param commits: Commits discovered for the branch, in any order.
    :param lower_bound: SHA of the inclusive compatibility floor.
    :return: Commits at or after the floor, oldest first.
    """

    ordered = sorted(commits, key=lambda commit: (commit.committed_at, commit.sha))
    for index, commit in enumerate(ordered):
        if commit.sha == lower_bound:
            return ordered[index:]
    raise UnsupportedHistoryError(
        f"Supported lower-bound commit {lower_bound} was not returned for the branch history."
    )


class SweepState:
    """The record of what has been dispatched, kept durable across interruptions."""

    def __init__(self, path: Path, commits: dict[str, dict[str, object]]) -> None:
        super().__init__()
        self.path = path
        self.commits = commits

    @classmethod
    def load(cls, path: Path) -> "SweepState":
        """Read the state file, treating a missing file as an empty sweep.

        A file that cannot be parsed is an error rather than something to
        overwrite: it may be the only record of hundreds of dispatched runs, and
        silently starting again would re-benchmark all of them.
        """

        if not path.exists():
            return cls(path, {})
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise StateFileError(f"{path} is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise StateFileError(f"{path} must contain a JSON object.")
        version = payload.get("version")
        if version != STATE_VERSION:
            raise StateFileError(f"{path} has unsupported state version {version!r}.")
        commits = payload.get("commits")
        if not isinstance(commits, dict):
            raise StateFileError(f"{path} lacks a commits object.")
        # A record that cannot be read is refused rather than dropped. Dropping it
        # would make the commit look untouched, and it would be dispatched again
        # even though its run may well exist.
        for sha, entry in commits.items():
            if not isinstance(entry, dict):
                raise StateFileError(f"{path} has a malformed record for {sha}.")
            if str(entry.get("status")) not in KNOWN_STATUSES:
                raise StateFileError(
                    f"{path} records unknown status {entry.get('status')!r} for {sha}."
                )
        return cls(path, {str(k): dict(v) for k, v in commits.items()})

    def save(self) -> None:
        """Write the state file atomically.

        Written to a sibling temporary file and renamed, so an interruption
        during the write leaves the previous state intact rather than a
        half-written file that the next run would refuse to load.
        """

        payload = {
            "version": STATE_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commits": self.commits,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def status(self, sha: str) -> Optional[str]:
        """Return the recorded status of one commit, or None when it is pending."""

        entry = self.commits.get(sha)
        return None if entry is None else str(entry.get("status"))

    def run_id(self, sha: str) -> Optional[int]:
        """Return the run recorded for one commit, when one has been recorded."""

        entry = self.commits.get(sha)
        if entry is None:
            return None
        value = entry.get("run_id")
        # bool is an int subclass, and True would otherwise be formatted into a
        # run endpoint as if it were an id.
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def record(self, sha: str, status: str, run_id: Optional[int] = None) -> None:
        """Set one commit's status and persist immediately."""

        entry: dict[str, object] = {"status": status}
        if run_id is not None:
            entry["run_id"] = run_id
        elif (existing := self.run_id(sha)) is not None:
            entry["run_id"] = existing
        self.commits[sha] = entry
        self.save()

    def counts(self) -> dict[str, int]:
        """Summarise the sweep for reporting."""

        tally = {SUCCEEDED: 0, FAILED: 0, IN_FLIGHT: 0, DISPATCHING: 0}
        for entry in self.commits.values():
            status = str(entry.get("status"))
            if status in tally:
                tally[status] += 1
        return tally


def conclude(run: WorkflowRun) -> Optional[str]:
    """Map a finished run onto a terminal commit status, or None while it runs."""

    if not run.finished:
        return None
    return SUCCEEDED if run.conclusion == "success" else FAILED


def sha_of(run: WorkflowRun) -> Optional[str]:
    """Extract the benchmarked commit from a run title.

    The workflow sets ``run-name`` to ``backfill-benchmark: <sha>`` precisely so
    that a run identifies its own commit. That is what lets a dispatch whose run
    id was never written be recovered instead of repeated.
    """

    _, separator, tail = run.display_title.partition("backfill-benchmark: ")
    if not separator:
        return None
    candidate = tail.strip().split()[0] if tail.strip() else ""
    return candidate or None


def reconcile(state: SweepState, github: GitHubCli, repository: str, workflow: str) -> None:
    """Resolve everything the previous process left unfinished.

    ``in_flight`` commits have a run id, so they are asked about directly.
    ``dispatching`` commits do not: the process died between the dispatch call
    and its state write, so the run may or may not exist. Those are matched
    against recent runs by commit, and only treated as pending when no run is
    found.
    """

    unfinished = [
        sha for sha, entry in state.commits.items() if str(entry.get("status")) not in TERMINAL
    ]
    if not unfinished:
        return

    for sha in list(unfinished):
        run_id = state.run_id(sha)
        if run_id is None:
            continue
        outcome = conclude(github.get_run(repository, run_id))
        if outcome is not None:
            state.record(sha, outcome, run_id)
            print(f"  {sha[:12]} finished while we were away: {outcome}")
        unfinished.remove(sha)

    orphans = [sha for sha in unfinished if state.status(sha) == DISPATCHING]
    if not orphans:
        return

    print(f"  {len(orphans)} commit(s) were mid-dispatch; searching recent runs for them")
    by_sha: dict[str, WorkflowRun] = {}
    for run in github.list_workflow_runs(repository, workflow):
        sha = sha_of(run)
        # Runs come back newest first, so the first sighting is the current one.
        if sha is not None and sha not in by_sha:
            by_sha[sha] = run
    for sha in orphans:
        run = by_sha.get(sha)
        if run is None:
            state.commits.pop(sha, None)
            state.save()
            print(f"  {sha[:12]} was never dispatched; it will be dispatched now")
            continue
        outcome = conclude(run)
        state.record(sha, outcome or IN_FLIGHT, run.run_id)
        print(f"  {sha[:12]} was dispatched as run {run.run_id}: {outcome or IN_FLIGHT}")


def dispatch(
    state: SweepState,
    github: GitHubCli,
    *,
    repository: str,
    workflow: str,
    workflow_ref: str,
    sha: str,
) -> bool:
    """Dispatch one commit, recording the intent before the call is made."""

    # Written first so that a crash during the dispatch leaves a marker to
    # reconcile against, rather than a commit that looks untouched.
    state.record(sha, DISPATCHING)
    try:
        result = github.dispatch_workflow(
            repository,
            workflow,
            workflow_ref=workflow_ref,
            inputs={"target_sha": sha},
        )
    except GitHubCliError as error:
        state.commits.pop(sha, None)
        state.save()
        print(f"  {sha[:12]} was not dispatched: {error}", file=sys.stderr)
        return False
    state.record(sha, IN_FLIGHT, result.run_id)
    print(f"  {sha[:12]} dispatched as run {result.run_id}")
    return True


def poll(state: SweepState, github: GitHubCli, repository: str) -> None:
    """Move every finished run out of flight.

    A run that cannot be asked about is left in flight and retried on the next
    pass. A sweep runs for hours, so a transient API failure has to cost one poll
    rather than the whole sweep.
    """

    for sha in [s for s, e in state.commits.items() if str(e.get("status")) == IN_FLIGHT]:
        run_id = state.run_id(sha)
        if run_id is None:
            continue
        try:
            outcome = conclude(github.get_run(repository, run_id))
        except GitHubCliError as error:
            print(f"  could not read run {run_id} for {sha[:12]}: {error}", file=sys.stderr)
            continue
        if outcome is not None:
            state.record(sha, outcome, run_id)
            print(f"  {sha[:12]} {outcome}")


def sweep(
    state: SweepState,
    github: GitHubCli,
    shas: Iterable[str],
    *,
    repository: str,
    workflow: str,
    workflow_ref: str,
    max_in_flight: int,
    poll_seconds: float,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Dispatch every pending commit, never exceeding the in-flight cap.

    Only a commit with no state entry is pending. Anything with an entry is
    already accounted for, and an ``in_flight`` one in particular has a run of
    its own that :func:`poll` is waiting on: dispatching it again would start a
    second run for the same commit and leave the first orphaned, since the state
    file keeps only the newer id. :func:`reconcile` is a precondition, and is
    what guarantees every entry here is either terminal or genuinely running.
    """

    remaining = [sha for sha in shas if state.status(sha) is None]
    consecutive_failures = 0
    while remaining or state.counts()[IN_FLIGHT]:
        poll(state, github, repository)

        while remaining and state.counts()[IN_FLIGHT] < max_in_flight:
            sha = remaining[0]
            if dispatch(
                state,
                github,
                repository=repository,
                workflow=workflow,
                workflow_ref=workflow_ref,
                sha=sha,
            ):
                remaining.pop(0)
                consecutive_failures = 0
            else:
                # A rejected dispatch is usually rate limiting or a transient API
                # failure, so leave it at the head of the queue and wait rather
                # than spinning through the rest of the sweep failing each one.
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_DISPATCH_FAILURES:
                    # Nothing is getting through, so this is a standing problem
                    # such as an expired token. Stop rather than log forever.
                    raise GitHubCliError(
                        f"{consecutive_failures} dispatches failed in a row; "
                        "check 'gh auth status' and re-run to resume."
                    )
                break

        if remaining or state.counts()[IN_FLIGHT]:
            tally = state.counts()
            print(
                f"{tally[SUCCEEDED]} succeeded, {tally[FAILED]} failed, "
                f"{tally[IN_FLIGHT]} in flight, {len(remaining)} pending",
                flush=True,
            )
            sleeper(poll_seconds)

    tally = state.counts()
    print(f"Sweep complete: {tally[SUCCEEDED]} succeeded, {tally[FAILED]} failed.")
    return 1 if tally[FAILED] else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--branch", default="main", help="Branch whose history is benchmarked.")
    parser.add_argument(
        "--workflow-ref",
        default="main",
        help="Ref the workflow definition is taken from.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        required=True,
        help="Where the sweep is recorded. Required: it is what makes an interrupted "
        "sweep resumable, and re-running without it would benchmark everything again.",
    )
    parser.add_argument("--max-in-flight", type=int, default=DEFAULT_MAX_IN_FLIGHT)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also re-dispatch commits recorded as failed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_in_flight < 1:
        print("--max-in-flight must be at least 1.", file=sys.stderr)
        return 1

    try:
        state = SweepState.load(args.state_file)
    except StateFileError as error:
        print(f"Refusing to start: {error}", file=sys.stderr)
        return 1

    try:
        github = GitHubCli()
        # Query from just before the floor's own timestamp. GitHub documents `since` as
        # "after the given time" while in practice returning commits at exactly that
        # instant, and the floor commit sits exactly on it. The SHA below is what actually
        # defines the boundary, so widening the query costs nothing.
        commits = github.list_commits(
            args.repository,
            args.branch,
            SUPPORTED_FLOOR_TIME - timedelta(seconds=1),
            datetime.now(timezone.utc),
        )
        supported = supported_commits(commits, SUPPORTED_FLOOR_SHA)

        if args.retry_failed:
            for sha, entry in list(state.commits.items()):
                if str(entry.get("status")) == FAILED:
                    state.commits.pop(sha)
            state.save()

        # Reconciliation decides what was already dispatched, so it cannot be skipped
        # when GitHub is unreachable: proceeding would dispatch those commits again.
        print(f"Reconciling {args.state_file} against GitHub")
        reconcile(state, github, args.repository, args.workflow)
    except (GitHubCliError, UnsupportedHistoryError) as error:
        print(f"Refusing to start: {error}", file=sys.stderr)
        return 1

    # Pending means the same thing here as it does in the sweep: no state entry at
    # all. Counting the in-flight ones as pending would report, and under --dry-run
    # promise, dispatches that will not happen because their runs already exist.
    pending = [c.sha for c in supported if state.status(c.sha) is None]
    tally = state.counts()
    print(
        f"{len(supported)} supported commit(s): {tally[SUCCEEDED]} succeeded, "
        f"{tally[FAILED]} failed, {tally[IN_FLIGHT]} already in flight, "
        f"{len(pending)} to dispatch, at most {args.max_in_flight} in flight."
    )
    if args.dry_run:
        for sha in pending:
            print(f"  would dispatch {sha[:12]}")
        return 0

    try:
        return sweep(
            state,
            github,
            pending,
            repository=args.repository,
            workflow=args.workflow,
            workflow_ref=args.workflow_ref,
            max_in_flight=args.max_in_flight,
            poll_seconds=args.poll_seconds,
        )
    except (KeyboardInterrupt, GitHubCliError) as error:
        tally = state.counts()
        detail = "Interrupted." if isinstance(error, KeyboardInterrupt) else f"Stopped: {error}"
        print(
            f"\n{detail} {tally[IN_FLIGHT]} run(s) left in flight and recorded in "
            f"{args.state_file}; re-run the same command to pick them up.",
            file=sys.stderr,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
