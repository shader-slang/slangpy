# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Split the supported commit history into batches and dispatch them all at once.

The batched workflow benchmarks a whole rev list per run, so the only thing left
to decide is how to cut the history up. There is no scheduler, no polling and no
in-flight cap: GitHub already queues dispatched runs and drains them as runners
free up, so every batch is submitted immediately and the queue does the rest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Optional, Sequence

try:
    from tools.benchmark_actions import Commit, GitHubCli, GitHubCliError
except ModuleNotFoundError:
    from benchmark_actions import Commit, GitHubCli, GitHubCliError

DEFAULT_REPOSITORY = "shader-slang/slangpy"
DEFAULT_WORKFLOW = "backfill-benchmark.yml"
DEFAULT_BATCH_SIZE = 30

# The oldest commit whose build the current benchmark harness can drive. Earlier
# commits are not a supported backfill target.
SUPPORTED_FLOOR_SHA = "f3ad0fd91d8cf4eeb2be3b505765b43482aa952a"
SUPPORTED_FLOOR_TIME = datetime(2025, 9, 2, 14, 42, 35, tzinfo=timezone.utc)


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

    Contiguous rather than strided: a batched sweep is expected to run to
    completion, so there is no need to sample the range evenly in case it is
    stopped early. The cost is that a partially finished sweep covers a set of
    chronological chunks instead of an even spread.

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
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


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
            # One rejected batch says nothing about the others, and the commits it
            # covers are named above, so keep going and report at the end.
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
    commits = github.list_commits(
        args.repository,
        args.branch,
        SUPPORTED_FLOOR_TIME,
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
