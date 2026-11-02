# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Decide which benchmarks a backfill target is too old to run.

The backfill builds a historical commit and then overlays today's benchmark harness
on top of it. A benchmark that did not exist in that commit cannot be measured
against it, and attempting to produces failures that are indistinguishable from
regressions. Ask git which benchmark modules are present in the target's tree and
list the ones that are not, so the run can skip exactly those.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def benchmark_modules(repo: Path, benchmark_dir: str) -> list[str]:
    """Repo-relative paths of the benchmark modules in the working tree.

    :param repo: Repository root.
    :param benchmark_dir: Repo-relative directory holding the benchmarks.
    :return: Sorted POSIX paths.
    """

    root = repo / benchmark_dir
    if not root.is_dir():
        return []
    return sorted(p.relative_to(repo).as_posix() for p in root.rglob("test_*.py"))


def exists_in_commit(repo: Path, commit: str, path: str) -> bool:
    """Whether a path is present in a commit's tree.

    :param repo: Repository root.
    :param commit: Commit to inspect.
    :param path: Repo-relative POSIX path.
    :return: True when the commit contains the path.
    """

    return (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}:{path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def benchmarks_postdating(repo: Path, commit: str, benchmark_dir: str) -> list[str]:
    """Benchmark modules that do not exist in a commit.

    :param repo: Repository root.
    :param commit: Backfill target commit.
    :param benchmark_dir: Repo-relative directory holding the benchmarks.
    :return: Sorted POSIX paths of modules the target predates.
    """

    return [
        path
        for path in benchmark_modules(repo, benchmark_dir)
        if not exists_in_commit(repo, commit, path)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-sha", required=True, help="Backfill target commit.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--benchmark-dir", default="slangpy/benchmarks", help="Directory holding benchmarks."
    )
    parser.add_argument("--output", type=Path, required=True, help="File to write the list to.")
    args = parser.parse_args()

    postdating = benchmarks_postdating(args.repo, args.target_sha, args.benchmark_dir)
    args.output.write_text("".join(f"{path}\n" for path in postdating), encoding="utf-8")

    if postdating:
        print(f"{len(postdating)} benchmark module(s) postdate {args.target_sha[:12]}:")
        for path in postdating:
            print(f"  {path}")
    else:
        print(f"All benchmark modules exist in {args.target_sha[:12]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
