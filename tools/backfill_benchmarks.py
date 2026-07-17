# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Safely dispatch bounded historical benchmark workflows with resumable state."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Literal, Optional, Sequence, TypeVar

try:
    from tools.benchmark_actions import (
        Commit,
        DispatchResult,
        GitHubCli,
        GitHubCliError,
        WorkflowRun,
    )
except ModuleNotFoundError:
    from benchmark_actions import Commit, DispatchResult, GitHubCli, GitHubCliError, WorkflowRun

DEFAULT_REPOSITORY = "shader-slang/slangpy"
DEFAULT_BRANCH = "main"
DEFAULT_WORKFLOW = "backfill-benchmark.yml"
DEFAULT_STATE_PATH = Path(".temp/benchmark-backfill-state.json")
SUPPORTED_FLOOR_SHA = "f3ad0fd91d8cf4eeb2be3b505765b43482aa952a"
SUPPORTED_FLOOR_TIME = datetime(2025, 9, 2, 14, 42, 35, tzinfo=timezone.utc)
STATE_SCHEMA_VERSION = 2
MAX_ACTIVE_RUNS = 4
DEFAULT_POLL_SECONDS = 60.0
DEFAULT_DISPATCH_GRACE = timedelta(minutes=10)

BackfillStatus = Literal["pending", "dispatching", "dispatched"]
ResultType = TypeVar("ResultType")


class BackfillStateError(RuntimeError):
    """Report corrupt or incompatible state without rewriting the source file."""


@dataclass
class BackfillRecord:
    """Track scheduling and returned workflow details for one supported commit."""

    sha: str
    committed_at: datetime
    message: str
    html_url: str
    status: BackfillStatus = "pending"
    dispatch_started_at: Optional[datetime] = None
    run_id: Optional[int] = None
    run_url: Optional[str] = None


@dataclass
class BackfillState:
    """Bind additive commit scheduling state to one exact backfill configuration."""

    schema_version: int
    repository: str
    branch: str
    workflow: str
    lower_bound: str
    records: dict[str, BackfillRecord]


def _utc_text(value: datetime) -> str:
    """Serialize an aware timestamp in stable RFC 3339 UTC form."""

    if value.tzinfo is None:
        raise BackfillStateError("Backfill timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any, field: str) -> datetime:
    """Parse a required state timestamp with a configuration-focused error."""

    if not isinstance(value, str):
        raise BackfillStateError(f"State field {field!r} must be a timestamp string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BackfillStateError(f"State field {field!r} is not a valid timestamp.") from error
    if parsed.tzinfo is None:
        raise BackfillStateError(f"State field {field!r} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _optional_time(value: Any, field: str) -> Optional[datetime]:
    """Parse an optional state timestamp while preserving an absent value."""

    return None if value is None else _parse_time(value, field)


def _required_string(value: Any, field: str) -> str:
    """Return one non-empty state string or identify the malformed field."""

    if not isinstance(value, str) or not value:
        raise BackfillStateError(f"State field {field!r} must be a non-empty string.")
    return value


def _new_state(repository: str, branch: str, workflow: str, lower_bound: str) -> BackfillState:
    """Create one empty state bound to the requested scheduler configuration."""

    return BackfillState(
        schema_version=STATE_SCHEMA_VERSION,
        repository=repository,
        branch=branch,
        workflow=workflow,
        lower_bound=lower_bound,
        records={},
    )


class BackfillStateStore:
    """Load and atomically replace one versioned local scheduler state file."""

    def __init__(
        self,
        path: Path,
        repository: str,
        branch: str,
        workflow: str,
        lower_bound: str,
    ) -> None:
        """Retain the exact configuration that every loaded state must match."""

        super().__init__()
        self.path = path
        self.repository = repository
        self.branch = branch
        self.workflow = workflow
        self.lower_bound = lower_bound

    def load(self) -> BackfillState:
        """Load compatible state or return a new in-memory state when absent."""

        if not self.path.exists():
            return _new_state(
                self.repository,
                self.branch,
                self.workflow,
                self.lower_bound,
            )
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            return self._parse_document(document)
        except (OSError, json.JSONDecodeError, BackfillStateError) as error:
            raise BackfillStateError(
                f"Cannot use backfill state {self.path}: {error}. Archive that file before "
                "starting a new scheduler; it was not modified."
            ) from error

    def _parse_document(self, document: Any) -> BackfillState:
        """Validate state identity and materialize its commit records."""

        if not isinstance(document, dict):
            raise BackfillStateError("state root must be a JSON object")
        expected = {
            "schemaVersion": STATE_SCHEMA_VERSION,
            "repository": self.repository,
            "branch": self.branch,
            "workflow": self.workflow,
            "lowerBound": self.lower_bound,
        }
        mismatches = [
            f"{field}={document.get(field)!r} (expected {value!r})"
            for field, value in expected.items()
            if document.get(field) != value
        ]
        if mismatches:
            raise BackfillStateError("incompatible state configuration: " + ", ".join(mismatches))
        raw_records = document.get("commits")
        if not isinstance(raw_records, list):
            raise BackfillStateError("state commits must be a JSON array")
        records: dict[str, BackfillRecord] = {}
        for index, raw_record in enumerate(raw_records):
            record = self._parse_record(raw_record, index)
            if record.sha in records:
                raise BackfillStateError(f"duplicate state commit {record.sha}")
            records[record.sha] = record
        return BackfillState(
            schema_version=STATE_SCHEMA_VERSION,
            repository=self.repository,
            branch=self.branch,
            workflow=self.workflow,
            lower_bound=self.lower_bound,
            records=records,
        )

    def _parse_record(self, value: Any, index: int) -> BackfillRecord:
        """Parse one commit record and enforce status-specific required fields."""

        if not isinstance(value, dict):
            raise BackfillStateError(f"state commit {index} must be a JSON object")
        status = value.get("status")
        if status not in ("pending", "dispatching", "dispatched"):
            raise BackfillStateError(f"state commit {index} has invalid status {status!r}")
        run_id = value.get("runId")
        if run_id is not None and (not isinstance(run_id, int) or isinstance(run_id, bool)):
            raise BackfillStateError(f"state commit {index} runId must be an integer or null")
        record = BackfillRecord(
            sha=_required_string(value.get("sha"), f"commits[{index}].sha"),
            committed_at=_parse_time(value.get("committedAt"), f"commits[{index}].committedAt"),
            message=_required_string(value.get("message"), f"commits[{index}].message"),
            html_url=_required_string(value.get("htmlUrl"), f"commits[{index}].htmlUrl"),
            status=status,
            dispatch_started_at=_optional_time(
                value.get("dispatchStartedAt"), f"commits[{index}].dispatchStartedAt"
            ),
            run_id=run_id,
            run_url=value.get("runUrl"),
        )
        if record.run_url is not None and not isinstance(record.run_url, str):
            raise BackfillStateError(f"state commit {index} runUrl must be a string or null")
        if record.status == "dispatching" and record.dispatch_started_at is None:
            raise BackfillStateError(f"state commit {index} dispatching status lacks a timestamp")
        if record.status == "dispatched" and (record.run_id is None or not record.run_url):
            raise BackfillStateError(f"state commit {index} dispatched status lacks run details")
        return record

    def save(self, state: BackfillState) -> None:
        """Flush a complete sibling file and atomically replace the visible state."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schemaVersion": state.schema_version,
            "repository": state.repository,
            "branch": state.branch,
            "workflow": state.workflow,
            "lowerBound": state.lower_bound,
            "commits": [
                {
                    "sha": record.sha,
                    "committedAt": _utc_text(record.committed_at),
                    "message": record.message,
                    "htmlUrl": record.html_url,
                    "status": record.status,
                    "dispatchStartedAt": (
                        _utc_text(record.dispatch_started_at)
                        if record.dispatch_started_at is not None
                        else None
                    ),
                    "runId": record.run_id,
                    "runUrl": record.run_url,
                }
                for record in sorted(
                    state.records.values(), key=lambda item: (item.committed_at, item.sha)
                )
            ],
        }
        data = (json.dumps(document, indent=2, sort_keys=False) + "\n").encode("utf-8")
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def backfill_run_title(sha: str) -> str:
    """Return the deterministic title used by the historical workflow."""

    return f"backfill-benchmark: {sha}"


def supported_commits(commits: Sequence[Commit], lower_bound: str) -> list[Commit]:
    """Return commits from the inclusive boundary forward in chronological order."""

    ordered = sorted(commits, key=lambda commit: (commit.committed_at, commit.sha))
    for index, commit in enumerate(ordered):
        if commit.sha == lower_bound:
            return ordered[index:]
    raise BackfillStateError(
        f"Supported lower-bound commit {lower_bound} was not returned for main history."
    )


def merge_discovered_commits(state: BackfillState, commits: Sequence[Commit]) -> bool:
    """Add newly discovered commits without changing any existing scheduling record."""

    changed = False
    for commit in commits:
        if commit.sha in state.records:
            continue
        state.records[commit.sha] = BackfillRecord(
            sha=commit.sha,
            committed_at=commit.committed_at,
            message=commit.message.splitlines()[0],
            html_url=commit.html_url,
        )
        changed = True
    return changed


def reconcile_records(
    state: BackfillState,
    runs: Sequence[WorkflowRun],
    now: datetime,
    grace: timedelta,
) -> bool:
    """Resolve deterministic titles and release expired uncertain dispatch markers."""

    matching_runs: dict[str, WorkflowRun] = {}
    for run in sorted(runs, key=lambda item: (item.created_at, item.run_id), reverse=True):
        if run.title.startswith("backfill-benchmark: "):
            matching_runs.setdefault(run.title, run)
    changed = False
    for record in state.records.values():
        matching = matching_runs.get(backfill_run_title(record.sha))
        if matching is not None:
            if (
                record.status != "dispatched"
                or record.run_id != matching.run_id
                or record.run_url != matching.html_url
            ):
                record.status = "dispatched"
                record.run_id = matching.run_id
                record.run_url = matching.html_url
                changed = True
            continue
        if (
            record.status == "dispatching"
            and record.dispatch_started_at is not None
            and now - record.dispatch_started_at >= grace
        ):
            record.status = "pending"
            record.dispatch_started_at = None
            changed = True
    return changed


def active_backfill_count(
    state: BackfillState,
    runs: Sequence[WorkflowRun],
    now: datetime,
    grace: timedelta,
) -> int:
    """Count active workflow runs plus recent dispatches not visible in GitHub yet."""

    listed_run_ids = {run.run_id for run in runs}
    active = sum(1 for run in runs if run.status != "completed")
    for record in state.records.values():
        if record.status not in ("dispatching", "dispatched"):
            continue
        if record.run_id is not None and record.run_id in listed_run_ids:
            continue
        if record.dispatch_started_at is not None and now - record.dispatch_started_at < grace:
            active += 1
    return active


def pending_records(state: BackfillState) -> list[BackfillRecord]:
    """Return pending commits in stable oldest-first dispatch order."""

    return sorted(
        (record for record in state.records.values() if record.status == "pending"),
        key=lambda record: (record.committed_at, record.sha),
    )


def incomplete_records(state: BackfillState) -> list[BackfillRecord]:
    """Return commits that are pending or awaiting uncertain-dispatch reconciliation."""

    return sorted(
        (record for record in state.records.values() if record.status != "dispatched"),
        key=lambda record: (record.committed_at, record.sha),
    )


def dispatch_cooldown_remaining(
    state: BackfillState,
    now: datetime,
    minimum_interval_seconds: float,
) -> float:
    """Return seconds before another dispatch is allowed across process restarts."""

    dispatch_times = [
        record.dispatch_started_at
        for record in state.records.values()
        if record.dispatch_started_at is not None
    ]
    if not dispatch_times:
        return 0.0
    elapsed = (now - max(dispatch_times)).total_seconds()
    return max(0.0, minimum_interval_seconds - elapsed)


def retry_read_operation(
    operation: Callable[[], ResultType],
    description: str,
    attempts: int = 3,
    delay_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> ResultType:
    """Retry read-only GitHub operations without duplicating workflow dispatches."""

    if attempts < 1:
        raise ValueError("Read retry attempts must be positive.")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except GitHubCliError:
            if attempt == attempts:
                raise
            print(f"{description} failed; retrying in {delay_seconds:g} second(s).")
            sleep(delay_seconds)
    raise AssertionError("unreachable read retry state")


def dispatch_oldest_pending(
    github: GitHubCli,
    store: BackfillStateStore,
    state: BackfillState,
    now: datetime,
    output: Callable[[str], None] = print,
) -> Optional[DispatchResult]:
    """Publish write-ahead state, dispatch one oldest commit, and save returned details."""

    pending = pending_records(state)
    if not pending:
        return None
    record = pending[0]
    record.status = "dispatching"
    record.dispatch_started_at = now
    record.run_id = None
    record.run_url = None
    store.save(state)
    result = github.dispatch_workflow(
        state.repository,
        state.workflow,
        workflow_ref=state.branch,
        inputs={"target_sha": record.sha},
    )
    record.status = "dispatched"
    record.run_id = result.run_id
    record.run_url = result.html_url
    store.save(state)
    output(f"Dispatched {record.sha}: {result.html_url}")
    return result


def run_backfill_scheduler(
    github: GitHubCli,
    store: BackfillStateStore,
    now_provider: Callable[[], datetime],
    sleep: Callable[[float], None],
    poll_seconds: float,
    once: bool,
    dry_run: bool,
    output: Callable[[str], None] = print,
) -> int:
    """Discover history, reconcile state, and dispatch at most one commit per interval."""

    if poll_seconds < 0:
        raise ValueError("Backfill poll interval must be non-negative.")
    now = now_provider().astimezone(timezone.utc)
    commits = retry_read_operation(
        lambda: github.list_commits(
            store.repository,
            store.branch,
            SUPPORTED_FLOOR_TIME,
            now,
        ),
        "Commit discovery",
        sleep=sleep,
    )
    supported = supported_commits(commits, store.lower_bound)
    state = store.load()
    merge_discovered_commits(state, supported)
    runs = retry_read_operation(
        lambda: github.list_workflow_runs(store.repository, store.workflow, maximum=1000),
        "Workflow reconciliation",
        sleep=sleep,
    )
    reconcile_records(state, runs, now, DEFAULT_DISPATCH_GRACE)
    output(
        f"Backfill {store.workflow} from {store.lower_bound}: {len(supported)} supported "
        f"commit(s), {len(runs)} existing run(s), {len(pending_records(state))} pending."
    )
    if dry_run:
        for record in pending_records(state):
            output(f"Would dispatch {store.workflow} for {record.sha} from {store.branch}.")
        output(
            f"Active backfill workflow count: "
            f"{active_backfill_count(state, runs, now, DEFAULT_DISPATCH_GRACE)}."
        )
        return 0
    store.save(state)
    while incomplete_records(state):
        now = now_provider().astimezone(timezone.utc)
        reconcile_records(state, runs, now, DEFAULT_DISPATCH_GRACE)
        store.save(state)
        if not incomplete_records(state):
            break
        active = active_backfill_count(state, runs, now, DEFAULT_DISPATCH_GRACE)
        output(f"Active backfill workflow count: {active}/{MAX_ACTIVE_RUNS}.")
        cooldown = dispatch_cooldown_remaining(state, now, poll_seconds)
        if active < MAX_ACTIVE_RUNS and cooldown <= 0:
            dispatch_oldest_pending(github, store, state, now, output)
        elif active < MAX_ACTIVE_RUNS:
            output(f"Next backfill dispatch is allowed in {cooldown:g} second(s).")
        if once:
            return 0
        if not incomplete_records(state):
            break
        sleep(poll_seconds)
        runs = retry_read_operation(
            lambda: github.list_workflow_runs(store.repository, store.workflow, maximum=100),
            "Workflow capacity poll",
            sleep=sleep,
        )
    output("All supported commits have been requested.")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Create the resumable backfill scheduler command-line parser."""

    parser = argparse.ArgumentParser(description="Dispatch bounded historical benchmark workflows.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the backfill scheduler with durable Ctrl+C and operator error behavior."""

    args = _parser().parse_args(argv)
    store = BackfillStateStore(
        args.state_file,
        repository=args.repository,
        branch=args.branch,
        workflow=args.workflow,
        lower_bound=SUPPORTED_FLOOR_SHA,
    )
    print(f"State file: {store.path}")
    print("Do not run another scheduler against this state file at the same time.")
    try:
        return run_backfill_scheduler(
            GitHubCli(),
            store,
            now_provider=lambda: datetime.now(timezone.utc),
            sleep=time.sleep,
            poll_seconds=DEFAULT_POLL_SECONDS,
            once=args.once,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("Backfill interrupted; durable state is ready for restart.", file=sys.stderr)
        return 130
    except (BackfillStateError, GitHubCliError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
