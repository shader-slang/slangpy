# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for auto-merge trigger functionality in the GitHub approval script.
"""

from unittest.mock import patch

# Import from the tools directory
import sys
import os

tools_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, tools_dir)

from gh_helpers import GitHubAPI


class TestAutoMergeTriggers:
    """Test auto-merge trigger detection functionality."""

    def test_auto_merge_trigger_in_description_basic(self):
        """Test basic [auto-merge] trigger in PR description."""
        github = GitHubAPI("fake-token")

        # Mock the get_pull_request method
        with patch.object(github, "get_pull_request") as mock_get_pr:
            mock_get_pr.return_value = {"body": "This PR should be auto-merged [auto-merge]"}

            # Mock the get_pull_request_comments method
            with patch.object(github, "get_pull_request_comments") as mock_get_comments:
                mock_get_comments.return_value = []

                result = github.check_auto_merge_trigger(123)
                assert result == "merge"

    def test_auto_merge_trigger_with_squash_method(self):
        """Test [auto-merge: squash] trigger in PR description."""
        github = GitHubAPI("fake-token")

        with patch.object(github, "get_pull_request") as mock_get_pr:
            mock_get_pr.return_value = {
                "body": "Please auto-merge this with squash [auto-merge: squash]"
            }

            with patch.object(github, "get_pull_request_comments") as mock_get_comments:
                mock_get_comments.return_value = []

                result = github.check_auto_merge_trigger(123)
                assert result == "squash"

    def test_auto_merge_trigger_with_rebase_method(self):
        """Test [auto-merge: rebase] trigger in PR description."""
        github = GitHubAPI("fake-token")

        with patch.object(github, "get_pull_request") as mock_get_pr:
            mock_get_pr.return_value = {"body": "Rebase and merge [auto-merge: rebase] when ready"}

            with patch.object(github, "get_pull_request_comments") as mock_get_comments:
                mock_get_comments.return_value = []

                result = github.check_auto_merge_trigger(123)
                assert result == "rebase"

    def test_auto_merge_trigger_in_comment(self):
        """Test auto-merge trigger in PR comments."""
        github = GitHubAPI("fake-token")

        with patch.object(github, "get_pull_request") as mock_get_pr:
            mock_get_pr.return_value = {"body": "Regular PR description"}

            with patch.object(github, "get_pull_request_comments") as mock_get_comments:
                mock_get_comments.return_value = [
                    {"body": "Some regular comment"},
                    {"body": "Ready to merge [auto-merge: squash]"},
                    {"body": "Another comment"},
                ]

                result = github.check_auto_merge_trigger(123)
                assert result == "squash"

    def test_auto_merge_trigger_case_insensitive(self):
        """Test auto-merge trigger is case insensitive."""
        github = GitHubAPI("fake-token")

        with patch.object(github, "get_pull_request") as mock_get_pr:
            mock_get_pr.return_value = {"body": "Please [AUTO-MERGE: SQUASH] this PR"}

            with patch.object(github, "get_pull_request_comments") as mock_get_comments:
                mock_get_comments.return_value = []

                result = github.check_auto_merge_trigger(123)
                assert result == "squash"

    def test_auto_merge_trigger_invalid_method(self):
        """Test auto-merge trigger with invalid method defaults to merge."""
        github = GitHubAPI("fake-token")

        with patch.object(github, "get_pull_request") as mock_get_pr:
            mock_get_pr.return_value = {"body": "Invalid method [auto-merge: invalid-method]"}

            with patch.object(github, "get_pull_request_comments") as mock_get_comments:
                mock_get_comments.return_value = []

                result = github.check_auto_merge_trigger(123)
                assert result == "squash"

    def test_auto_merge_no_trigger_found(self):
        """Test when no auto-merge trigger is found."""
        github = GitHubAPI("fake-token")

        with patch.object(github, "get_pull_request") as mock_get_pr:
            mock_get_pr.return_value = {"body": "Regular PR description with no triggers"}

            with patch.object(github, "get_pull_request_comments") as mock_get_comments:
                mock_get_comments.return_value = [
                    {"body": "Regular comment"},
                    {"body": "Another comment without triggers"},
                ]

                result = github.check_auto_merge_trigger(123)
                assert result is None

    def test_auto_merge_trigger_whitespace_handling(self):
        """Test auto-merge trigger handles whitespace properly."""
        github = GitHubAPI("fake-token")

        with patch.object(github, "get_pull_request") as mock_get_pr:
            mock_get_pr.return_value = {"body": "Whitespace test [auto-merge:  squash  ]"}

            with patch.object(github, "get_pull_request_comments") as mock_get_comments:
                mock_get_comments.return_value = []

                result = github.check_auto_merge_trigger(123)
                assert result == "squash"

    def test_auto_merge_multiple_triggers_first_wins(self):
        """Test when multiple auto-merge triggers exist, first one wins."""
        github = GitHubAPI("fake-token")

        with patch.object(github, "get_pull_request") as mock_get_pr:
            mock_get_pr.return_value = {
                "body": "First trigger [auto-merge: squash] and second [auto-merge: rebase]"
            }

            with patch.object(github, "get_pull_request_comments") as mock_get_comments:
                mock_get_comments.return_value = []

                result = github.check_auto_merge_trigger(123)
                assert result == "squash"
