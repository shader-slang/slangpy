# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed GitHub Actions operations implemented through the official ``gh`` CLI."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import shutil
import subprocess
from typing import Any, Optional, Protocol, Sequence

GITHUB_API_VERSION = "2026-03-10"


class GitHubCliError(RuntimeError):
    """Report an unavailable, unauthenticated, or malformed GitHub CLI operation."""


class CommandRunner(Protocol):
    """Describe the injectable subprocess boundary used by :class:`GitHubCli`."""

    def __call__(
        self, arguments: Sequence[str], input_text: Optional[str]
    ) -> subprocess.CompletedProcess[str]:
        """Execute one argument-array command and return its captured result."""

        ...


@dataclass(frozen=True)
class Commit:
    """Describe one commit reachable from the selected repository branch."""

    sha: str
    committed_at: datetime
    message: str
    html_url: str


@dataclass(frozen=True)
class WorkflowRun:
    """Describe the GitHub fields needed for title reconciliation and capacity."""

    run_id: int
    title: str
    status: str
    conclusion: Optional[str]
    html_url: str
    head_sha: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DispatchResult:
    """Describe the workflow run returned directly by a successful dispatch."""

    run_id: int
    run_url: str
    html_url: str


def _default_command_runner(
    arguments: Sequence[str], input_text: Optional[str]
) -> subprocess.CompletedProcess[str]:
    """Run ``gh`` without a shell while capturing deterministic UTF-8 text output."""

    return subprocess.run(
        list(arguments),
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        shell=False,
    )


def _utc_text(value: datetime) -> str:
    """Serialize an aware timestamp in the RFC 3339 UTC spelling expected by GitHub."""

    if value.tzinfo is None:
        raise ValueError("GitHub time bounds must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any, field: str) -> datetime:
    """Parse one required GitHub timestamp and identify malformed response fields."""

    if not isinstance(value, str):
        raise GitHubCliError(f"GitHub response field {field!r} must be a timestamp string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GitHubCliError(
            f"GitHub response field {field!r} is not a valid timestamp."
        ) from error
    if parsed.tzinfo is None:
        raise GitHubCliError(f"GitHub response field {field!r} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _required_string(value: Any, field: str) -> str:
    """Return a required non-empty GitHub string or raise a response-shape error."""

    if not isinstance(value, str) or not value:
        raise GitHubCliError(f"GitHub response field {field!r} must be a non-empty string.")
    return value


def _flatten_pages(value: Any, collection_field: Optional[str]) -> list[Any]:
    """Flatten the page arrays emitted by ``gh api --paginate --slurp``."""

    if not isinstance(value, list):
        raise GitHubCliError("GitHub CLI pagination output must be a JSON array.")
    flattened: list[Any] = []
    for page in value:
        collection = page
        if collection_field is not None:
            if not isinstance(page, dict):
                raise GitHubCliError("GitHub workflow-run page must be a JSON object.")
            collection = page.get(collection_field)
        if not isinstance(collection, list):
            raise GitHubCliError("GitHub CLI page does not contain the expected JSON array.")
        flattened.extend(collection)
    return flattened


class GitHubCli:
    """Expose the GitHub operations needed by benchmark scheduling through ``gh``."""

    def __init__(
        self,
        command_runner: Optional[CommandRunner] = None,
        executable: Optional[str] = None,
    ) -> None:
        """Locate ``gh`` and retain an injectable no-shell command boundary."""

        super().__init__()
        resolved = executable if executable is not None else shutil.which("gh")
        if resolved is None:
            raise GitHubCliError(
                "GitHub CLI 'gh' was not found. Install it and run 'gh auth login' before retrying."
            )
        self._executable = resolved
        self._command_runner = command_runner or _default_command_runner

    def _invoke(self, arguments: Sequence[str], input_text: Optional[str] = None) -> Any:
        """Execute one versioned ``gh api`` call and decode its JSON response."""

        command = [
            self._executable,
            "api",
            "--header",
            f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
            *arguments,
        ]
        result = self._command_runner(command, input_text)
        if result.returncode != 0:
            diagnostic = (result.stderr or result.stdout or "unknown error").strip()[:2048]
            raise GitHubCliError(
                f"GitHub CLI request failed with exit code {result.returncode}: {diagnostic}. "
                "Run 'gh auth status' to verify authentication."
            )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise GitHubCliError("GitHub CLI returned invalid JSON.") from error

    def list_commits(
        self,
        repository: str,
        branch: str,
        since: datetime,
        until: datetime,
    ) -> list[Commit]:
        """List branch commits in the inclusive committer-time interval."""

        if until < since:
            raise ValueError("GitHub commit interval end must not precede its start.")
        response = self._invoke(
            [
                "--method",
                "GET",
                "--paginate",
                "--slurp",
                f"repos/{repository}/commits",
                "-f",
                f"sha={branch}",
                "-f",
                f"since={_utc_text(since)}",
                "-f",
                f"until={_utc_text(until)}",
                "-f",
                "per_page=100",
            ]
        )
        commits: list[Commit] = []
        for index, item in enumerate(_flatten_pages(response, None)):
            if not isinstance(item, dict):
                raise GitHubCliError(f"GitHub commit at index {index} must be a JSON object.")
            commit = item.get("commit")
            if not isinstance(commit, dict):
                raise GitHubCliError(f"GitHub commit at index {index} lacks commit data.")
            committer = commit.get("committer")
            if not isinstance(committer, dict):
                raise GitHubCliError(f"GitHub commit at index {index} lacks committer data.")
            commits.append(
                Commit(
                    sha=_required_string(item.get("sha"), "sha"),
                    committed_at=_parse_datetime(committer.get("date"), "commit.committer.date"),
                    message=_required_string(commit.get("message"), "commit.message"),
                    html_url=_required_string(item.get("html_url"), "html_url"),
                )
            )
        return commits

    def list_workflow_runs(self, repository: str, workflow: str, maximum: int) -> list[WorkflowRun]:
        """List up to ``maximum`` newest runs for one workflow definition."""

        if maximum < 1:
            raise ValueError("Workflow run maximum must be positive.")
        per_page = min(maximum, 100)
        runs: list[WorkflowRun] = []
        page = 1
        while len(runs) < maximum:
            response = self._invoke(
                [
                    "--method",
                    "GET",
                    f"repos/{repository}/actions/workflows/{workflow}/runs",
                    "-f",
                    f"per_page={per_page}",
                    "-f",
                    f"page={page}",
                ]
            )
            if not isinstance(response, dict):
                raise GitHubCliError("GitHub workflow-run page must be a JSON object.")
            items = response.get("workflow_runs")
            if not isinstance(items, list):
                raise GitHubCliError("GitHub workflow-run page lacks the workflow_runs JSON array.")
            for item in items:
                index = len(runs)
                if not isinstance(item, dict):
                    raise GitHubCliError(f"GitHub workflow run at index {index} must be an object.")
                run_id = item.get("id")
                conclusion = item.get("conclusion")
                if not isinstance(run_id, int) or isinstance(run_id, bool):
                    raise GitHubCliError("GitHub workflow run id must be an integer.")
                if conclusion is not None and not isinstance(conclusion, str):
                    raise GitHubCliError("GitHub workflow run conclusion must be a string or null.")
                runs.append(
                    WorkflowRun(
                        run_id=run_id,
                        title=_required_string(item.get("display_title"), "display_title"),
                        status=_required_string(item.get("status"), "status"),
                        conclusion=conclusion,
                        html_url=_required_string(item.get("html_url"), "html_url"),
                        head_sha=_required_string(item.get("head_sha"), "head_sha"),
                        created_at=_parse_datetime(item.get("created_at"), "created_at"),
                        updated_at=_parse_datetime(item.get("updated_at"), "updated_at"),
                    )
                )
                if len(runs) == maximum:
                    break
            if len(items) < per_page or len(runs) == maximum:
                break
            page += 1
        return runs

    def dispatch_workflow(
        self,
        repository: str,
        workflow: str,
        workflow_ref: str,
        inputs: dict[str, str],
    ) -> DispatchResult:
        """Dispatch one workflow and return GitHub's immediate run identity and URLs."""

        body = json.dumps(
            {
                "ref": workflow_ref,
                "inputs": inputs,
                "return_run_details": True,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        response = self._invoke(
            [
                "--method",
                "POST",
                "--input",
                "-",
                f"repos/{repository}/actions/workflows/{workflow}/dispatches",
            ],
            body,
        )
        if not isinstance(response, dict):
            raise GitHubCliError("GitHub workflow dispatch response must be a JSON object.")
        run_id = response.get("workflow_run_id")
        if not isinstance(run_id, int) or isinstance(run_id, bool):
            raise GitHubCliError("GitHub workflow dispatch response lacks workflow_run_id.")
        return DispatchResult(
            run_id=run_id,
            run_url=_required_string(response.get("run_url"), "run_url"),
            html_url=_required_string(response.get("html_url"), "html_url"),
        )
