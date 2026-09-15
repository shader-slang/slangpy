# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Split the supported commit history into batches and dispatch them all at once.

The batched workflow benchmarks a whole rev list per run, so the only thing left
to decide is how to cut the history up. There is no scheduler, no polling and no
in-flight cap: GitHub already queues dispatched runs and drains them as runners
free up, so every batch is submitted immediately and the queue does the rest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Optional, Sequence

# Importable both as ``tools.dispatch_backfill_batches`` and as a directly executed
# script, which only puts tools/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.backfill_batch import SUPPORTED_FLOOR_SHA, SUPPORTED_FLOOR_TIME
from tools.benchmark_actions import Commit, GitHubCli, GitHubCliError

DEFAULT_REPOSITORY = "shader-slang/slangpy"
DEFAULT_WORKFLOW = "backfill-benchmark.yml"
# Sized from measured Windows per-commit cost so that a worst-case batch leaves ample
# headroom against the workflow timeout. Larger batches buy little: the queue admission
# is already amortised away, and with only a couple of Windows performance runners,
# fewer and larger batches divide the work less evenly between them.
DEFAULT_BATCH_SIZE = 45


class UnsupportedHistoryError(RuntimeError):
    """The discovered history does not contain the configured compatibility floor."""


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


def batch_commits(commits: Sequence[Commit], size: int) -> list[list[Commit]]:
    """Cut chronologically ordered commits into contiguous batches.

    Batches are contiguous rather than strided, so a partially finished sweep
    covers a set of chronological chunks rather than an even spread of the range.

    :param commits: Commits, oldest first.
    :param size: Commits per batch.
    :return: Batches in chronological order.
    """

    if size < 1:
        raise ValueError("Batch size must be positive.")
    return [list(commits[start : start + size]) for start in range(0, len(commits), size)]


def batch_label(batch: Sequence[Commit], index: int, total: int) -> str:
    """Describe a batch compactly enough to read in the workflow run list."""

    first = batch[0].committed_at.strftime("%Y-%m-%d")
    last = batch[-1].committed_at.strftime("%Y-%m-%d")
    return f"batch {index}/{total} ({len(batch)} commits, {first}..{last})"


def read_exclusions(path: Optional[Path]) -> set[str]:
    """Read commits to leave out, one per line, ignoring blanks and comments."""

    if path is None:
        return set()
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def dispatch_batches(
    github: GitHubCli,
    *,
    repository: str,
    workflow: str,
    workflow_ref: str,
    batches: Sequence[Sequence[Commit]],
    dry_run: bool,
) -> int:
    """Dispatch every batch and report the run each one became."""

    failures = 0
    for index, batch in enumerate(batches, start=1):
        label = batch_label(batch, index, len(batches))
        if dry_run:
            print(f"Would dispatch {label}")
            continue
        try:
            result = github.dispatch_workflow(
                repository,
                workflow,
                workflow_ref=workflow_ref,
                inputs={
                    "target_shas": " ".join(commit.sha for commit in batch),
                    "batch_label": label,
                },
            )
        except GitHubCliError as error:
            # One rejected batch says nothing about the others, so keep going and
            # report the total at the end.
            print(f"{label} was not dispatched: {error}", file=sys.stderr)
            failures += 1
            continue
        print(f"Dispatched {label}: {result.html_url}")
    return failures


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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--exclude-file",
        type=Path,
        help="File of commits to leave out, one per line, e.g. ones already benchmarked.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    github = GitHubCli()
    # Query from just before the floor's own timestamp. GitHub documents `since` as
    # "after the given time" while in practice returning commits at exactly that
    # instant, and the floor commit sits exactly on it. The SHA below is what actually
    # defines the boundary, so widening the query costs nothing and removes any
    # dependence on how that ambiguity is resolved.
    commits = github.list_commits(
        args.repository,
        args.branch,
        SUPPORTED_FLOOR_TIME - timedelta(seconds=1),
        datetime.now(timezone.utc),
    )
    supported = supported_commits(commits, SUPPORTED_FLOOR_SHA)

    excluded = read_exclusions(args.exclude_file)
    selected = [commit for commit in supported if commit.sha not in excluded]
    batches = batch_commits(selected, args.batch_size)
    print(
        f"{len(supported)} supported commit(s), {len(supported) - len(selected)} excluded, "
        f"{len(selected)} to benchmark in {len(batches)} batch(es) of up to {args.batch_size}."
    )

    failures = dispatch_batches(
        github,
        repository=args.repository,
        workflow=args.workflow,
        workflow_ref=args.workflow_ref,
        batches=batches,
        dry_run=args.dry_run,
    )
    if failures:
        print(f"{failures} batch(es) were not dispatched.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
