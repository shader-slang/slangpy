# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Simple test for CI trigger requirement functionality.
"""

import sys
import os
from typing import Any

# Import from the tools directory
tools_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, tools_dir)

from gh_approve import approve_pr


def test_ci_environment_requires_trigger_flag(monkeypatch: Any) -> None:
    """Test that CI environment returns error when require_trigger is False."""

    # Mock the CI environment detection to return True
    monkeypatch.setattr("gh_approve.is_running_in_ci", lambda: True)

    # Mock the GitHub API class to avoid actual API calls
    class MockGitHub:
        def __init__(
            self, token: str, repo_owner: str = "shader-slang", repo_name: str = "slangpy"
        ) -> None:
            super().__init__()
            self.token = token
            self.repo_owner = repo_owner
            self.repo_name = repo_name

        def get_pull_request(self, pr_number: int) -> dict[str, Any]:
            return {
                "user": {"login": "testuser"},
                "title": "Test PR",
                "body": "Test PR body without triggers",
            }

    monkeypatch.setattr("gh_approve.GitHubAPI", MockGitHub)

    # Test: CI environment without require_trigger should return error
    result = approve_pr(
        pr_number=123,
        token="fake-token",
        approved_users="testuser",
        require_trigger=False,  # This should cause an error in CI
        dry_run=True,
    )

    assert result == "error", "Should return error when in CI without require_trigger"
