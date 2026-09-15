# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Offline coverage for the GitHub adapter and the ordinary benchmark workflow."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Optional, Sequence

import yaml

from tools import benchmark_actions as actions

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class FakeCommandRunner:
    """Return queued ``gh`` results while retaining exact arguments and stdin."""

    def __init__(self, responses: Sequence[subprocess.CompletedProcess[str]]) -> None:
        """Copy deterministic responses so each invocation consumes one result."""

        super().__init__()
        self.responses = list(responses)
        self.calls: list[tuple[list[str], Optional[str]]] = []

    def __call__(
        self, arguments: Sequence[str], input_text: Optional[str]
    ) -> subprocess.CompletedProcess[str]:
        """Capture one no-shell invocation and return its next queued response."""

        self.calls.append((list(arguments), input_text))
        if not self.responses:
            raise AssertionError("Unexpected GitHub CLI invocation.")
        return self.responses.pop(0)


def completed_process(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    """Create one captured subprocess result for a fake command runner."""

    return subprocess.CompletedProcess(
        args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_the_gh_adapter_flattens_pages_and_sends_inputs_on_stdin() -> None:
    """``--slurp`` returns an array of pages, and inputs never reach the command line."""

    first_sha, second_sha = "1" * 40, "2" * 40
    pages = [
        [
            {
                "sha": sha,
                "html_url": f"https://github.test/commit/{sha}",
                "commit": {"message": "m", "committer": {"date": "2026-07-16T10:00:00Z"}},
            }
        ]
        for sha in (first_sha, second_sha)
    ]
    dispatch = {
        "workflow_run_id": 43,
        "run_url": "https://api.github.test/runs/43",
        "html_url": "https://github.test/runs/43",
    }
    runner = FakeCommandRunner(
        [completed_process(json.dumps(pages)), completed_process(json.dumps(dispatch))]
    )
    github = actions.GitHubCli(command_runner=runner, executable="gh-test")

    commit_list = github.list_commits(
        "shader-slang/slangpy",
        "main",
        datetime(2026, 7, 16, tzinfo=timezone.utc),
        datetime(2026, 7, 17, tzinfo=timezone.utc),
    )
    result = github.dispatch_workflow(
        "shader-slang/slangpy",
        "ci-benchmark.yml",
        workflow_ref="main",
        inputs={"revision": "a" * 40},
    )

    assert [c.sha for c in commit_list] == [first_sha, second_sha]
    assert "sha=main" in runner.calls[0][0]
    assert result.run_id == 43
    assert json.loads(runner.calls[1][1] or "") == {
        "ref": "main",
        "inputs": {"revision": "a" * 40},
        "return_run_details": True,
    }


def test_the_ordinary_workflow_benchmarks_only_a_validated_revision() -> None:
    """The job carries the BenchView write key and then runs the revision's own code.

    Checking out the raw input would let a branch or tag move between validation and
    checkout, and validating only the explicit input would leave the dispatch default
    able to benchmark an unmerged commit.
    """

    text = (REPOSITORY_ROOT / ".github/workflows/ci-benchmark.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    # YAML 1.1 reads an unquoted `on` as the boolean true, which is how GitHub spells
    # the trigger block.
    workflow["on"] = workflow.pop(True)
    resolved = "${{ needs.validate-revision.outputs.revision }}"

    build = workflow["jobs"]["build"]
    assert build["needs"] == "validate-revision"
    checkout = next(s for s in build["steps"] if "checkout" in s.get("uses", ""))
    assert checkout["with"]["ref"] == resolved
    assert build["env"]["BENCHVIEW_BENCHMARK_REF"] == resolved

    resolve = next(
        s for s in workflow["jobs"]["validate-revision"]["steps"] if s.get("id") == "resolve"
    )
    assert resolve["env"]["DEFAULT_REVISION"] == "${{ github.sha }}"
    assert 'requested="${REVISION:-$DEFAULT_REVISION}"' in resolve["run"]
    # One resolution and one ancestry check, so neither path can bypass the other.
    assert resolve["run"].count("git rev-parse") == 1
    assert resolve["run"].count("merge-base --is-ancestor") == 1
