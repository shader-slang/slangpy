# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test GitHub API integration functionality"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gh_helpers import GitHubAPI, get_github_token, get_repo_info, get_pr_number_from_ci


class TestGitHubAPI:
    """Test GitHub API functionality"""

    def test_github_api_initialization(self):
        """Test that GitHubAPI can be initialized with parameters"""
        api = GitHubAPI("fake_token", "test_owner", "test_repo")

        assert api.token == "fake_token"
        assert api.repo_owner == "test_owner"
        assert api.repo_name == "test_repo"
        assert api.base_url == "https://api.github.com"

    def test_github_api_headers_contain_token(self):
        """Test that API headers contain authorization token"""
        api = GitHubAPI("test_token_123", "owner", "repo")

        assert "Authorization" in api.headers
        assert (
            api.headers["Authorization"] == "token test_token_123"
        )  # GitHub uses "token" not "Bearer"
        assert api.headers["Accept"] == "application/vnd.github.v3+json"

    def test_get_github_token_from_env(self, fake_github_token: str):
        """Test getting GitHub token from environment variable"""
        token = get_github_token()
        assert token == "fake_token_for_testing"

    def test_get_repo_info_from_env(self, clean_environment: None):
        """Test getting repository info from environment variables"""
        # Test with full repository string
        os.environ["GITHUB_REPOSITORY"] = "test_owner/test_repo"

        owner, name = get_repo_info()
        assert owner == "test_owner"
        assert name == "test_repo"

        # Test with separate owner and name
        del os.environ["GITHUB_REPOSITORY"]
        os.environ["GITHUB_REPOSITORY_OWNER"] = "separate_owner"
        os.environ["GITHUB_REPOSITORY_NAME"] = "separate_repo"

        owner, name = get_repo_info()
        assert owner == "separate_owner"
        assert name == "separate_repo"

    def test_get_pr_number_from_ci_github_ref(self, clean_environment: None):
        """Test extracting PR number from GitHub Actions environment"""
        # Test with GITHUB_REF
        os.environ["GITHUB_REF"] = "refs/pull/123/merge"
        pr_number = get_pr_number_from_ci()
        assert pr_number == 123

        # Test with invalid GITHUB_REF
        os.environ["GITHUB_REF"] = "refs/heads/main"
        pr_number = get_pr_number_from_ci()
        assert pr_number is None
