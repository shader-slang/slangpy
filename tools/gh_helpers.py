# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
GitHub API helper functions for interacting with repositories and pull requests.
"""

import os
import requests
from typing import Dict, Any, Optional, List


class GitHubAPI:
    """Helper class for GitHub API interactions."""

    def __init__(self, token: str, repo_owner: str = "shader-slang", repo_name: str = "slangpy"):
        """
        Initialize GitHub API client.

        Args:
            token: GitHub personal access token
            repo_owner: Repository owner (default: shader-slang)
            repo_name: Repository name (default: slangpy)
        """
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "slangpy-gh-tools",
        }

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        """
        Make a request to the GitHub API.

        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            endpoint: API endpoint (without base URL)
            data: Optional data to send with request

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        if method.upper() == "GET":
            response = requests.get(url, headers=self.headers, params=data)
        elif method.upper() == "POST":
            response = requests.post(url, headers=self.headers, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=self.headers, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=self.headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            # For 422 errors, try to get more detailed error information
            if response.status_code == 422:
                try:
                    error_data = response.json()
                    if "errors" in error_data and error_data["errors"]:
                        error_msg = "; ".join(error_data["errors"])
                        raise requests.HTTPError(
                            f"{e.response.status_code} Client Error: {error_msg} for url: {e.response.url}"
                        ) from e
                except:
                    pass
            raise
        return response

    def get_pull_request(self, pr_number: int) -> Dict[str, Any]:
        """
        Get details about a pull request.

        Args:
            pr_number: Pull request number

        Returns:
            Dictionary containing PR details
        """
        endpoint = f"/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}"
        response = self._make_request("GET", endpoint)
        return response.json()

    def approve_pull_request(self, pr_number: int, comment: str = "") -> Dict[str, Any]:
        """
        Approve a pull request.

        Args:
            pr_number: Pull request number
            comment: Optional comment to add with approval

        Returns:
            Dictionary containing review details
        """
        endpoint = f"/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}/reviews"
        data = {"event": "APPROVE"}
        if comment:
            data["body"] = comment

        response = self._make_request("POST", endpoint, data)
        return response.json()

    def list_pull_request_reviews(self, pr_number: int) -> List[Dict[str, Any]]:
        """
        List all reviews for a pull request.

        Args:
            pr_number: Pull request number

        Returns:
            List of review dictionaries
        """
        endpoint = f"/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}/reviews"
        response = self._make_request("GET", endpoint)
        return response.json()

    def get_pull_request_comments(self, pr_number: int) -> List[Dict[str, Any]]:
        """
        Get all comments on a pull request.

        Args:
            pr_number: Pull request number

        Returns:
            List of comment dictionaries
        """
        endpoint = f"/repos/{self.repo_owner}/{self.repo_name}/issues/{pr_number}/comments"
        response = self._make_request("GET", endpoint)
        return response.json()

    def check_approval_trigger(
        self, pr_number: int, trigger_pattern: str = r"\[auto-approve:([^\]]*)\]"
    ) -> Optional[str]:
        """
        Check if PR description or comments contain an approval trigger.

        Args:
            pr_number: Pull request number
            trigger_pattern: Regex pattern to match approval trigger (default: [auto-approve: comment])

        Returns:
            The approval comment from the trigger, or None if not found
        """
        import re

        # Get PR details
        pr_info = self.get_pull_request(pr_number)

        # Check PR description
        if pr_info.get("body"):
            match = re.search(trigger_pattern, pr_info["body"], re.IGNORECASE)
            if match:
                return (
                    match.group(1).strip()
                    if match.groups()
                    else "Auto-approved from PR description"
                )

        # Check PR comments
        comments = self.get_pull_request_comments(pr_number)
        for comment in comments:
            if comment.get("body"):
                match = re.search(trigger_pattern, comment["body"], re.IGNORECASE)
                if match:
                    return (
                        match.group(1).strip()
                        if match.groups()
                        else "Auto-approved from PR comment"
                    )

        return None


def check_approved_user(
    username: str, approved_users: Optional[str] = None, approved_users_env: str = "APPROVED_USERS"
) -> bool:
    """
    Check if a username is in the approved users list.

    Args:
        username: GitHub username to check
        approved_users: Comma-separated string of approved users (takes precedence over env var)
        approved_users_env: Environment variable name containing comma-separated approved users

    Returns:
        True if user is approved or no restriction is set, False otherwise

    Raises:
        ValueError: If running in CI and no approved users list is configured
    """
    # Use provided approved_users string, or fall back to environment variable
    approved_users_str = approved_users or os.environ.get(approved_users_env)

    # If running in CI, approved users list is mandatory
    if is_running_in_ci() and not approved_users_str:
        raise ValueError(
            "Approved users list is required when running in CI environment. "
            f"Set {approved_users_env} environment variable or use --approved-users parameter."
        )

    # If no approved users list is set (and not in CI), allow all users
    if not approved_users_str:
        return True

    # Parse comma-separated list and normalize (strip whitespace, lowercase)
    approved_users_list = [
        user.strip().lower() for user in approved_users_str.split(",") if user.strip()
    ]

    # Check if user is in the approved list (case insensitive)
    return username.lower() in approved_users_list


def is_running_in_ci() -> bool:
    """
    Check if the script is running in a CI environment.

    Returns:
        True if running in CI (GitHub Actions), False otherwise
    """
    return os.environ.get("CI") == "true"


def get_github_token() -> str:
    """
    Get GitHub token from environment variable.

    Returns:
        GitHub token

    Raises:
        ValueError: If token is not found
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable not set")
    return token


def get_repo_info() -> tuple[str, str]:
    """
    Get repository owner and name from environment variables or defaults.

    Returns:
        Tuple of (owner, repo_name)
    """
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "shader-slang")
    repo_name = os.environ.get("GITHUB_REPOSITORY_NAME", "slangpy")

    # Handle the case where GITHUB_REPOSITORY is set (format: owner/repo)
    github_repo = os.environ.get("GITHUB_REPOSITORY")
    if github_repo and "/" in github_repo:
        owner, repo_name = github_repo.split("/", 1)

    return owner, repo_name


def get_pr_number_from_ci() -> Optional[int]:
    """
    Get PR number from GitHub Actions environment variables.

    Returns:
        PR number if found, None otherwise
    """
    # GitHub Actions sets these environment variables for PR events
    pr_number = None

    # Method 1: Direct PR number from GitHub event
    github_pr = os.environ.get("GITHUB_PR_NUMBER")
    if github_pr:
        try:
            return int(github_pr)
        except ValueError:
            pass

    # Method 2: Extract from GITHUB_REF for pull request events
    # Format: refs/pull/{pr_number}/merge or refs/pull/{pr_number}/head
    github_ref = os.environ.get("GITHUB_REF")
    if github_ref and github_ref.startswith("refs/pull/"):
        try:
            pr_part = github_ref.split("/")[2]  # refs/pull/{number}/merge
            return int(pr_part)
        except (IndexError, ValueError):
            pass

    # Method 3: GitHub event payload (for pull_request events)
    github_event_name = os.environ.get("GITHUB_EVENT_NAME")
    if github_event_name == "pull_request":
        # In GitHub Actions, the PR number is often available as GITHUB_EVENT_NUMBER
        github_event_number = os.environ.get("GITHUB_EVENT_NUMBER")
        if github_event_number:
            try:
                return int(github_event_number)
            except ValueError:
                pass

    return None
