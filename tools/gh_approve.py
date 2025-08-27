#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
GitHub Pull Request Approval Tool

This script allows approving GitHub pull requests either via command line arguments
or environment variables (for CI usage). When run in GitHub Actions, it can
auto-detect the PR number from the CI environment.

Supports approval triggers for security - requiring specific text in PR description
or comments before allowing approval (e.g., [auto-approve: Ready to merge]).

Usage:
    # Manual usage with PR number
    python tools/gh_approve.py --pr=<pr_number> [--token=<github_token>] [--comment=<comment>]

    # CI usage (PR number auto-detected)
    python tools/gh_approve.py [--comment=<comment>]

    # Require approval trigger for security
    python tools/gh_approve.py --require-trigger

    # Custom trigger pattern
    python tools/gh_approve.py --require-trigger --trigger-pattern="\\[approve-me:([^\\]]*)\\]"

Environment Variables:
    GITHUB_TOKEN: GitHub personal access token
    GITHUB_REPOSITORY: Repository in format owner/repo (optional, defaults to shader-slang/slangpy)
    GITHUB_REPOSITORY_OWNER: Repository owner (optional, defaults to shader-slang)
    APPROVED_USERS: Comma-separated list of GitHub usernames allowed to use auto-approval (optional, defaults to allowing all users)
                    NOTE: Required when running in CI environment (CI=true)
    GITHUB_REPOSITORY_NAME: Repository name (optional, defaults to slangpy)

    # Auto-detected in GitHub Actions:
    GITHUB_REF: Used to extract PR number (refs/pull/{number}/merge)
    GITHUB_EVENT_NAME: Used to detect pull_request events
    GITHUB_PR_NUMBER: Direct PR number (if available)
    GITHUB_EVENT_NUMBER: Alternative PR number source

Approval Triggers:
    Default pattern: [auto-approve: <comment>]
    Behavior:
    - Trigger found: Proceeds with approval
    - No trigger: Skips approval gracefully (exit code 0)
    - Error: Only for actual failures (exit code 1)
    Examples:
    - [auto-approve: Ready to merge]
    - [auto-approve: Dependency update]
    - [auto-approve: Minor fix]
"""

import argparse
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gh_helpers import (
    GitHubAPI,
    get_github_token,
    get_repo_info,
    get_pr_number_from_ci,
    check_approved_user,
    is_running_in_ci,
)


# Set up logger
logger = logging.getLogger(__name__)


def approve_pr(
    pr_number: int,
    token: str,
    comment: Optional[str] = None,
    repo_owner: str = "shader-slang",
    repo_name: str = "slangpy",
    require_trigger: bool = False,
    trigger_pattern: str = r"\[auto-approve:([^\]]*)\]",
    approved_users: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """
    Approve a pull request.

    Args:
        pr_number: Pull request number to approve
        token: GitHub personal access token
        comment: Optional comment to add with approval
        repo_owner: Repository owner
        repo_name: Repository name
        require_trigger: Whether to require approval trigger in PR/comments
        trigger_pattern: Regex pattern for approval trigger
        approved_users: Comma-separated list of approved users (overrides APPROVED_USERS env var)
        dry_run: If True, perform all checks but skip the actual approval

    Returns:
        "approved": Approval was successful
        "skipped": Approval was skipped (e.g., missing trigger)
        "error": An error occurred
    """
    try:
        github = GitHubAPI(token, repo_owner, repo_name)

        if dry_run:
            logger.info("Dry run mode - performing all checks but will not actually approve")

        # First, get PR details to verify it exists
        logger.debug(f"Fetching PR #{pr_number} details...")
        pr_info = github.get_pull_request(pr_number)
        logger.debug(f"PR Title: {pr_info['title']}")
        logger.debug(f"PR Author: {pr_info['user']['login']}")
        logger.debug(f"PR State: {pr_info['state']}")

        if pr_info["state"] != "open":
            logger.warning(f"PR #{pr_number} is not open (state: {pr_info['state']})")
            return "error"

        # Check if PR author is in approved users list
        pr_author = pr_info["user"]["login"]

        # Add CI detection info
        if is_running_in_ci():
            logger.debug("Running in CI environment - approved users list is mandatory")

        try:
            if not check_approved_user(pr_author, approved_users):
                logger.info(f"User '{pr_author}' is not in the approved users list")
                logger.debug(
                    f"Auto-approval is restricted to users listed in APPROVED_USERS environment variable"
                )
                logger.info("Auto-approval skipped")
                return "skipped"
        except ValueError as e:
            logger.error(str(e))
            return "error"

        # Check for approval trigger if required
        if require_trigger:
            logger.debug("Checking for approval trigger...")
            trigger_comment = github.check_approval_trigger(pr_number, trigger_pattern)
            if not trigger_comment:
                logger.info("No approval trigger found in PR description or comments")
                logger.debug(f"Expected pattern: {trigger_pattern}")
                logger.info("Auto-approval skipped. Add '[auto-approve: comment]' to enable")
                return "skipped"  # Return skipped - this is not an error condition
            else:
                logger.debug(f"Approval trigger found: '{trigger_comment}'")
                # Use the trigger comment if no explicit comment was provided
                if not comment:
                    comment = trigger_comment

        # Check if already approved by current user
        reviews = github.list_pull_request_reviews(pr_number)
        current_user_reviews = [r for r in reviews if r["state"] == "APPROVED"]

        if current_user_reviews:
            logger.debug(f"PR #{pr_number} already has approvals")

        # Approve the PR (or simulate in dry-run mode)
        approval_comment = comment or f"Approved via gh_approve.py tool"

        if dry_run:
            logger.info(
                f"DRY RUN: Would approve PR #{pr_number} with comment: '{approval_comment}'"
            )
            logger.info("Dry run completed - all checks passed, would proceed with approval")
            return "approved"
        else:
            logger.debug(f"Approving PR #{pr_number}...")
            review = github.approve_pull_request(pr_number, approval_comment)

            logger.info(f"Successfully approved PR #{pr_number}")
            logger.debug(f"Review ID: {review['id']}")
            if comment:
                logger.debug(f"Comment: {comment}")

            return "approved"

    except Exception as e:
        error_str = str(e)

        # Check for specific GitHub errors
        if "422 Client Error" in error_str and "Unprocessable Entity" in error_str:
            # This is likely the "can't approve your own PR" error
            logger.error(
                f"Cannot approve PR #{pr_number}: You cannot approve your own pull request"
            )
            logger.debug(f"PR Author: {pr_info['user']['login']}")
            logger.debug("This is a GitHub policy restriction.")
            logger.debug("Ask another team member to approve this PR.")
        elif "404 Client Error" in error_str:
            logger.error(f"PR #{pr_number} not found")
            logger.debug("Check that the PR number is correct and the repository is accessible")
        elif "403 Client Error" in error_str:
            logger.error("Access denied")
            logger.debug("Check that your GitHub token has the necessary permissions")
        else:
            logger.error(f"Error approving PR #{pr_number}: {error_str}")
        return "error"


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Approve GitHub Pull Requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--pr",
        type=int,
        help="Pull request number to approve (auto-detected in CI if not provided)",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub personal access token (default: GITHUB_TOKEN env var)",
    )

    parser.add_argument("--comment", type=str, help="Optional comment to add with approval")

    parser.add_argument(
        "--repo-owner", type=str, help="Repository owner (default: from env vars or shader-slang)"
    )

    parser.add_argument(
        "--repo-name", type=str, help="Repository name (default: from env vars or slangpy)"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without actually approving"
    )

    parser.add_argument(
        "--require-trigger",
        action="store_true",
        help="Require approval trigger in PR description or comments (e.g., [auto-approve: comment])",
    )

    parser.add_argument(
        "--trigger-pattern",
        type=str,
        default=r"\[auto-approve:([^\]]*)\]",
        help="Regex pattern for approval trigger (default: [auto-approve: comment])",
    )

    parser.add_argument(
        "--approved-users",
        type=str,
        default=os.environ.get("APPROVED_USERS"),
        help="Comma-separated list of GitHub usernames allowed to use auto-approval (default: APPROVED_USERS env var, or allow all users if not set)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (show debug information)",
    )

    args = parser.parse_args()

    # Set up logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Early validation for CI environment requirements
    if is_running_in_ci():
        if not args.approved_users and not os.environ.get("APPROVED_USERS"):
            logger.error("Running in CI environment - approved users list is mandatory")
            logger.error("Approved users list is required when running in CI environment.")
            logger.error(
                "Set APPROVED_USERS environment variable or use --approved-users parameter."
            )
            sys.exit(1)
        else:
            logger.debug("Running in CI environment - approved users list is mandatory")

    # Get PR number - either from argument or auto-detect from CI
    pr_number = args.pr
    if not pr_number:
        pr_number = get_pr_number_from_ci()
        if pr_number:
            logger.debug(f"Auto-detected PR number from CI: {pr_number}")
        else:
            logger.error("No PR number provided and could not auto-detect from CI")
            logger.error("Please provide --pr argument or run in GitHub Actions PR context")
            sys.exit(1)

    # Get token
    if not args.token:
        try:
            args.token = get_github_token()
        except ValueError as e:
            logger.error(str(e))
            logger.error("Please provide --token argument or set GITHUB_TOKEN environment variable")
            sys.exit(1)

    # Get repository info
    if not args.repo_owner or not args.repo_name:
        env_owner, env_name = get_repo_info()
        repo_owner = args.repo_owner or env_owner
        repo_name = args.repo_name or env_name
    else:
        repo_owner = args.repo_owner
        repo_name = args.repo_name

    logger.debug(f"Repository: {repo_owner}/{repo_name}")
    logger.debug(f"PR Number: {pr_number}")

    # Call approve_pr with dry_run parameter
    result = approve_pr(
        pr_number,
        args.token,
        args.comment,
        repo_owner,
        repo_name,
        args.require_trigger,
        args.trigger_pattern,
        args.approved_users,
        args.dry_run,
    )

    if result == "error":
        sys.exit(1)
    elif result == "skipped":
        logger.info("Approval process completed - PR was not approved due to missing trigger")
        sys.exit(0)  # Exit successfully since this is expected behavior
    elif result == "approved":
        logger.info("Approval process completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
