# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Benchmark one historical commit inside one CI job.

One commit per job makes the unit of work, the unit of failure and the unit of
retry the same thing: a runner that dies costs only the commit it was holding,
and retrying that commit cannot re-run one that already succeeded. The
dispatcher in ``backfill_benchmarks.py`` is what keeps the resulting fleet of
runs bounded and resumable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Optional, Sequence

# The oldest commit whose build the current benchmark harness can drive. Earlier
# commits are not a supported backfill target and are rejected before any work
# starts. The timestamp is only used to bound the commit query in the dispatcher.
#
# It sits just past the last target needing an era-dependent accommodation, so
# the harness carries none. Lowering it means reinstating them.
SUPPORTED_FLOOR_SHA = "5c266df695fe69da052ef2495e55f9d571442e23"
SUPPORTED_FLOOR_TIME = datetime(2026, 2, 16, 16, 4, 51, tzinfo=timezone.utc)

# The harness must be identical at every commit or the timings are not
# comparable, so these paths are copied over the historical tree. slangpy/benchmarks
# is included both because older benchmarks predate device isolation and because
# the benchmark definitions themselves have to be held fixed.
OVERLAY_PATHS = (
    "tools/ci.py",
    "tools/gpu_clock.py",
    "tools/backfill_benchmark_manifest.py",
    "slangpy/testing/benchmark",
    "slangpy/testing/helpers.py",
    "slangpy/testing/plugin.py",
    "slangpy/testing/crashpad.py",
    "slangpy/benchmarks",
)


class CommitFailure(RuntimeError):
    """A stage failed, so this commit produced no measurement."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage


def run(
    command: Sequence[str], cwd: Path, stage: str, env: Optional[dict[str, str]] = None
) -> None:
    """Run a command, streaming its output, and raise :class:`CommitFailure` on error."""

    print(f"+ {' '.join(command)}", flush=True)
    completed = subprocess.run(list(command), cwd=str(cwd), env=env, check=False)
    if completed.returncode != 0:
        raise CommitFailure(stage, f"exited with code {completed.returncode}")


def attempt(command: Sequence[str], cwd: Optional[Path] = None) -> bool:
    """Run a command whose failure is a recoverable outcome rather than an error."""

    print(f"+ {' '.join(command)}", flush=True)
    completed = subprocess.run(list(command), cwd=None if cwd is None else str(cwd), check=False)
    return completed.returncode == 0


def is_ancestor(repo: Path, candidate: str, descendant: str, stage: str) -> bool:
    """Answer whether ``candidate`` precedes ``descendant``.

    ``git merge-base --is-ancestor`` exits 1 for a negative answer and something else
    for a question it could not answer at all, such as a revision that does not
    resolve. Only 0 and 1 are answers; anything else is raised as ``stage`` rather
    than silently read as "no".
    """

    command = ["git", "merge-base", "--is-ancestor", candidate, descendant]
    print(f"+ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(repo), check=False)
    if completed.returncode in (0, 1):
        return completed.returncode == 0
    raise CommitFailure(
        stage,
        f"could not decide whether {candidate} precedes {descendant} "
        f"(git exited with code {completed.returncode})",
    )


def _retry_writable(function: Callable[[str], Any], path: str, _excinfo: Any) -> None:
    """Retry a removal that failed because Git marks its object files read-only."""

    os.chmod(path, stat.S_IWRITE)
    function(path)


def remove_tree(path: Path) -> None:
    """Delete a directory tree, including the read-only files Git leaves on Windows."""

    if not path.exists():
        return
    print(f"Removing {path}", flush=True)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry_writable)
    else:
        shutil.rmtree(path, onerror=_retry_writable)


def prepare_clone(work_dir: Path, repository_url: str) -> Path:
    """Create a pristine clone of the repository under ``work_dir``.

    The clone path is derived here rather than supplied, so the directory removed
    is always one this function chose. Any earlier clone is removed first: the
    runners keep their working directories between jobs, so a cancelled or killed
    run can leave one behind, and building on top of it would silently benchmark a
    tree nobody chose.
    """

    clone = work_dir / "clone"
    remove_tree(clone)
    work_dir.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--recursive", repository_url, str(clone)], work_dir, "clone")
    return clone


def reset_to_commit(clone: Path, sha: str) -> None:
    """Return the historical clone to a pristine checkout of one commit.

    The overlay writes untracked files and replaces tracked ones, and the build
    leaves artifacts behind, so the tree is reset and cleaned rather than merely
    checked out. The build only takes about a minute with a warm ccache, which
    makes a fully clean tree cheaper than reasoning about contamination.
    """

    run(["git", "reset", "--hard"], clone, "reset")
    run(["git", "clean", "-xfd"], clone, "reset")
    run(["git", "checkout", "--detach", sha], clone, "checkout")
    run(["git", "submodule", "sync", "--recursive"], clone, "submodules")
    run(["git", "submodule", "update", "--init", "--recursive", "--force"], clone, "submodules")
    run(["git", "lfs", "pull"], clone, "lfs")


def overlay_harness(workspace: Path, clone: Path) -> None:
    """Copy the current benchmark harness over the historical tree."""

    for relative in OVERLAY_PATHS:
        source = workspace / relative
        destination = clone / relative
        if not source.exists():
            raise CommitFailure("overlay", f"harness is missing {relative}")
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    print(f"Overlaid {len(OVERLAY_PATHS)} harness path(s) from {workspace}")


def benchmark_commit(
    sha: str,
    *,
    workspace: Path,
    clone: Path,
    skip_manifest: Path,
    run_id: str,
    api_url: str,
) -> None:
    """Build one historical commit and submit today's benchmarks against it."""

    python = sys.executable
    environment = dict(os.environ)
    environment["BACKFILL_TARGET_SHA"] = sha
    environment["BENCHVIEW_BENCHMARK_REF"] = sha
    environment["BACKFILL_SKIP_BENCHMARKS"] = str(skip_manifest)

    reset_to_commit(clone, sha)
    run([python, "-m", "pip", "install", "-r", "requirements-dev.txt"], clone, "pip")
    run([python, "-m", "pip", "install", "-r", "samples/requirements.txt"], clone, "pip")
    run([python, "tools/ci.py", "setup"], clone, "setup")
    # Old sources are compiled by whatever toolchain the runner has now, which
    # emits warnings that did not exist when they were written. The point is to
    # measure historical performance, not to re-validate warning cleanliness.
    run(
        [python, "tools/ci.py", "--cmake-args=-DSGL_WARNINGS_AS_ERRORS=OFF", "configure"],
        clone,
        "configure",
    )
    run([python, "tools/ci.py", "build"], clone, "build")
    overlay_harness(workspace, clone)
    # The overlay brings back benchmarks that did not exist in the target. Running
    # those measures nothing and fails in ways that look like regressions, so list
    # them for the harness to skip.
    run(
        [
            python,
            "tools/backfill_benchmark_manifest.py",
            "--target-sha",
            sha,
            "--output",
            str(skip_manifest),
        ],
        clone,
        "manifest",
    )
    run([python, "tools/ci.py", "install-slangpy-torch"], clone, "torch-bridge", environment)
    try:
        run(
            [
                python,
                "tools/ci.py",
                "benchmark-python",
                "--run-id",
                run_id,
                "--api-url",
                api_url,
                "--lock-gpu-clocks",
            ],
            clone,
            "benchmark",
            environment,
        )
    finally:
        # The runners are persistent, so a stale bridge would be imported by the
        # next commit's benchmarks.
        attempt([python, "-m", "pip", "uninstall", "slangpy-torch", "-y"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-sha",
        help="Commit to benchmark. Defaults to $BACKFILL_TARGET_SHA, which is how CI passes "
        "it: keeping it out of the command line means the same workflow step works verbatim "
        "under both pwsh and bash.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Scratch directory. The clone and the skip list are placed inside it, and "
        "the clone is removed before and after the run.",
    )
    parser.add_argument(
        "--repository-url",
        default="https://github.com/shader-slang/slangpy.git",
        help="Repository to clone the historical source from.",
    )
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Harness checkout root.")
    parser.add_argument("--run-id", required=True, help="BenchView run identifier.")
    parser.add_argument("--api-url", required=True, help="BenchView API URL.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    sha = (args.target_sha or os.environ.get("BACKFILL_TARGET_SHA", "")).strip()
    if not sha:
        print("No commit was requested.", file=sys.stderr)
        return 1

    work_dir = args.work_dir.resolve()
    workspace = args.workspace.resolve()

    started = time.monotonic()
    clone = None
    try:
        clone = prepare_clone(work_dir, args.repository_url)
        # Ancestry can only be resolved once the history is on disk, and the harness
        # checkout is shallow, so the floor is checked after the clone rather than
        # before it. The dispatcher already refuses to dispatch below the floor; this
        # is the backstop for a hand-run job.
        if not is_ancestor(clone, SUPPORTED_FLOOR_SHA, sha, "floor"):
            print(
                f"{sha[:12]} predates the supported floor {SUPPORTED_FLOOR_SHA[:12]}.",
                file=sys.stderr,
            )
            return 1
        benchmark_commit(
            sha,
            workspace=workspace,
            clone=clone,
            skip_manifest=work_dir / "skip-benchmarks.txt",
            run_id=args.run_id,
            api_url=args.api_url,
        )
    except CommitFailure as failure:
        # The dispatcher reads the run conclusion, not this text, but the commit is
        # named here so the log says which one failed without opening the summary.
        print(f"::error::{sha} failed at {failure.stage}: {failure}", flush=True)
        print(f"{sha} failed after {(time.monotonic() - started) / 60:.1f} min", file=sys.stderr)
        return 1
    finally:
        # A build tree is tens of gigabytes and the runners are not guaranteed to be
        # discarded, so it is removed whatever the outcome.
        if clone is not None:
            remove_tree(clone)

    print(f"{sha} succeeded in {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
