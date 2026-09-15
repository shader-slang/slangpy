# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Benchmark a list of historical commits inside a single CI job.

One workflow run per commit spends roughly 80% of its wall time waiting for a
self-hosted runner, so this driver pays that admission once and then walks a
whole rev list on the runner it was given.

Each commit is isolated: a failure is recorded against that commit and the loop
moves on. Results reach BenchView as the loop proceeds, so a job that is killed
or times out keeps everything already submitted, and the commits it never
reached are named in the summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Optional, Sequence

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
    """A stage failed for one commit; the batch continues with the next."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage


@dataclass
class CommitOutcome:
    """The result recorded for one commit of the batch."""

    sha: str
    status: str
    stage: Optional[str]
    seconds: float


def parse_rev_list(text: str) -> list[str]:
    """Split a whitespace- or comma-separated rev list into commits.

    :param text: Raw workflow input.
    :return: Commits in the given order, duplicates removed.
    """

    seen: dict[str, None] = {}
    for token in text.replace(",", " ").split():
        seen.setdefault(token, None)
    return list(seen)


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


def git_output(repo: Path, *arguments: str) -> str:
    """Return the stripped stdout of a git command that must succeed."""

    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stdout.strip()


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


def repin_vcpkg(clone: Path, harness_vcpkg: str) -> None:
    """Move a vcpkg pin older than the harness one forward to the harness one.

    MSYS2 deletes superseded packages, so old vcpkg revisions request an
    msys2-runtime that no longer exists, cannot bootstrap pkgconf, and so cannot
    build at all. The harness revision is the newest one the backfill knows to work,
    and any pin strictly older than it is at risk, so those are moved forward.

    A pin that is *not* an ancestor of the harness one is left alone. It is either the
    same revision or a newer one, and the dependency set is part of what the backfill
    measures, so moving it would change the thing being timed.

    Ancestry is resolved against the harness revision itself rather than a hardcoded
    boundary commit. An earlier version compared against a constant that held a
    *slangpy* SHA rather than a vcpkg one; since that object does not exist in the
    vcpkg repository the test could never succeed, every pin looked fine, and the ten
    oldest commits in the range failed at configure instead of being repaired.

    This has to run after ``ci.py setup``, which resets every submodule to its
    recorded revision.
    """

    target_vcpkg = git_output(clone, "rev-parse", "HEAD:external/vcpkg")
    if target_vcpkg == harness_vcpkg:
        print(f"Target already pins the harness vcpkg revision {target_vcpkg}")
        return

    vcpkg = clone / "external" / "vcpkg"
    # The submodule clone is not guaranteed to carry the harness revision, and an
    # ancestry test against a missing object fails the same way as a negative result.
    if not attempt(["git", "cat-file", "-e", f"{harness_vcpkg}^{{commit}}"], vcpkg):
        if not attempt(["git", "fetch", "--no-tags", "origin", harness_vcpkg], vcpkg):
            run(["git", "fetch", "--no-tags", "origin"], vcpkg, "vcpkg")

    # Both objects are present, so this now answers the ancestry question rather than
    # reporting that it could not be asked.
    if not attempt(["git", "merge-base", "--is-ancestor", target_vcpkg, harness_vcpkg], vcpkg):
        print(f"Keeping the target vcpkg revision {target_vcpkg}, which is not older")
        return

    print(f"Replacing unbootstrappable vcpkg {target_vcpkg} with {harness_vcpkg}")
    run(["git", "checkout", "--detach", harness_vcpkg], vcpkg, "vcpkg")


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
    harness_vcpkg: str,
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
    repin_vcpkg(clone, harness_vcpkg)
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


def format_report(outcomes: Sequence[CommitOutcome], pending: Sequence[str]) -> str:
    """Render the batch result as a Markdown table plus explicit commit lists."""

    lines = ["| commit | result | stage | minutes |", "|---|---|---|---|"]
    for outcome in outcomes:
        lines.append(
            f"| `{outcome.sha[:12]}` | {outcome.status} | {outcome.stage or ''} "
            f"| {outcome.seconds / 60:.1f} |"
        )
    for sha in pending:
        lines.append(f"| `{sha[:12]}` | not reached | | |")

    succeeded = [o.sha for o in outcomes if o.status == "success"]
    failed = [o.sha for o in outcomes if o.status == "failed"]
    lines.append("")
    lines.append(
        f"**{len(succeeded)} succeeded, {len(failed)} failed, {len(pending)} not reached.**"
    )
    for label, shas in (("Succeeded", succeeded), ("Failed", failed), ("Not reached", pending)):
        if shas:
            lines.append("")
            lines.append(f"{label}: `{' '.join(shas)}`")
    return "\n".join(lines) + "\n"


def run_batch(
    shas: Sequence[str],
    *,
    workspace: Path,
    clone: Path,
    skip_manifest: Path,
    run_id: str,
    api_url: str,
    summary_path: Optional[Path],
) -> int:
    """Benchmark every commit in the batch and report what happened.

    :return: Process exit code; non-zero only if every commit failed.
    """

    harness_vcpkg = git_output(workspace, "rev-parse", "HEAD:external/vcpkg")
    outcomes: list[CommitOutcome] = []

    def publish(echo: bool) -> None:
        report = format_report(outcomes, shas[len(outcomes) :])
        if summary_path is not None:
            summary_path.write_text(report, encoding="utf-8")
        if echo:
            print(report, flush=True)

    for index, sha in enumerate(shas, start=1):
        print(f"\n::group::[{index}/{len(shas)}] {sha}", flush=True)
        started = time.monotonic()
        try:
            benchmark_commit(
                sha,
                workspace=workspace,
                clone=clone,
                harness_vcpkg=harness_vcpkg,
                skip_manifest=skip_manifest,
                run_id=run_id,
                api_url=api_url,
            )
            outcome = CommitOutcome(sha, "success", None, time.monotonic() - started)
        except CommitFailure as failure:
            print(f"::error::{sha} failed at {failure.stage}: {failure}", flush=True)
            outcome = CommitOutcome(sha, "failed", failure.stage, time.monotonic() - started)
        print("::endgroup::", flush=True)
        outcomes.append(outcome)
        # The file is rewritten after every commit so a job that is killed still
        # reports the work it managed to submit. Echoing the whole table that often
        # would reprint every row once per commit, so that is left until the end.
        publish(echo=False)

    publish(echo=True)
    return 1 if outcomes and all(o.status == "failed" for o in outcomes) else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-shas",
        help="Whitespace- or comma-separated commits. Defaults to $BACKFILL_TARGET_SHAS, which "
        "is how CI passes the list: keeping it out of the command line means the same workflow "
        "step works verbatim under both pwsh and bash.",
    )
    parser.add_argument("--clone-dir", type=Path, required=True, help="Historical clone root.")
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Harness checkout root.")
    parser.add_argument("--floor-sha", required=True, help="Oldest supported commit.")
    parser.add_argument("--skip-manifest", type=Path, required=True, help="Skip-list scratch file.")
    parser.add_argument("--run-id", required=True, help="BenchView run identifier.")
    parser.add_argument("--api-url", required=True, help="BenchView API URL.")
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Markdown report destination. Defaults to $GITHUB_STEP_SUMMARY.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    shas = parse_rev_list(args.target_shas or os.environ.get("BACKFILL_TARGET_SHAS", ""))
    if not shas:
        print("No commits were requested.", file=sys.stderr)
        return 1
    summary_path = args.summary_file
    if summary_path is None and os.environ.get("GITHUB_STEP_SUMMARY"):
        summary_path = Path(os.environ["GITHUB_STEP_SUMMARY"])

    clone = args.clone_dir.resolve()
    workspace = args.workspace.resolve()

    # Reject the whole batch up front rather than discovering an unsupported
    # commit part-way through a multi-hour run.
    unsupported = [
        sha
        for sha in shas
        if subprocess.run(
            ["git", "-C", str(clone), "merge-base", "--is-ancestor", args.floor_sha, sha],
            check=False,
        ).returncode
        != 0
    ]
    if unsupported:
        print(
            f"{len(unsupported)} commit(s) predate the supported floor {args.floor_sha[:12]}: "
            + " ".join(sha[:12] for sha in unsupported),
            file=sys.stderr,
        )
        return 1

    print(f"Benchmarking {len(shas)} commit(s) in one job.")
    return run_batch(
        shas,
        workspace=workspace,
        clone=clone,
        skip_manifest=args.skip_manifest,
        run_id=args.run_id,
        api_url=args.api_url,
        summary_path=summary_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
