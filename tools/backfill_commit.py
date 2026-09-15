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
import subprocess
import sys
import time
from typing import Optional, Sequence

# The oldest commit whose build the current benchmark harness can drive. Earlier
# commits are not a supported backfill target and are rejected before any work
# starts. The timestamp is only used to bound the commit query in the dispatcher.
#
# This is "Updated vcpkg to enable the new USD (#489)", the commit that moved
# external/vcpkg to the revision every later commit still pins. The ten commits
# below it pin a vcpkg old enough that MSYS2 has deleted the packages it asks
# for, so they cannot bootstrap and cannot be built at all.
SUPPORTED_FLOOR_SHA = "dba4ce185fc05836f0dab86ed8e41977c84673ac"
SUPPORTED_FLOOR_TIME = datetime(2025, 9, 5, 10, 49, 2, tzinfo=timezone.utc)

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
    parser.add_argument("--clone-dir", type=Path, required=True, help="Historical clone root.")
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Harness checkout root.")
    parser.add_argument("--skip-manifest", type=Path, required=True, help="Skip-list scratch file.")
    parser.add_argument("--run-id", required=True, help="BenchView run identifier.")
    parser.add_argument("--api-url", required=True, help="BenchView API URL.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    sha = (args.target_sha or os.environ.get("BACKFILL_TARGET_SHA", "")).strip()
    if not sha:
        print("No commit was requested.", file=sys.stderr)
        return 1

    clone = args.clone_dir.resolve()
    workspace = args.workspace.resolve()

    try:
        if not is_ancestor(clone, SUPPORTED_FLOOR_SHA, sha, "floor"):
            print(
                f"{sha[:12]} predates the supported floor {SUPPORTED_FLOOR_SHA[:12]}.",
                file=sys.stderr,
            )
            return 1
    except CommitFailure as failure:
        print(failure, file=sys.stderr)
        return 1

    started = time.monotonic()
    try:
        benchmark_commit(
            sha,
            workspace=workspace,
            clone=clone,
            skip_manifest=args.skip_manifest,
            run_id=args.run_id,
            api_url=args.api_url,
        )
    except CommitFailure as failure:
        # The dispatcher reads the run conclusion, not this text, but the commit is
        # named here so the log says which one failed without opening the summary.
        print(f"::error::{sha} failed at {failure.stage}: {failure}", flush=True)
        print(f"{sha} failed after {(time.monotonic() - started) / 60:.1f} min", file=sys.stderr)
        return 1

    print(f"{sha} succeeded in {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
