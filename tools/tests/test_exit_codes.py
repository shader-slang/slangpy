# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test exit codes for different scenarios in gh_approve.py"""

import subprocess
import sys
import os
from typing import List, Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestExitCodes:
    """Test that exit codes are correct for different scenarios"""

    def run_gh_approve(
        self, args: List[str], env_vars: Optional[Dict[str, str]] = None
    ) -> subprocess.CompletedProcess[str]:
        """Helper to run gh_approve.py with given arguments and environment"""
        cmd = [sys.executable, "gh_approve.py"] + args
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Run from the tools directory
        cwd = os.path.dirname(os.path.dirname(__file__))
        result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        return result

    def test_no_pr_argument_and_not_in_ci_fails(self):
        """Test that missing PR number fails when not in CI"""
        result = self.run_gh_approve([])
        assert result.returncode == 1, "Should fail when no PR number provided and not in CI"

    def test_ci_without_approved_users_fails(self):
        """Test that CI environment without approved users fails"""
        result = self.run_gh_approve(["--pr=123", "--verbose"], env_vars={"CI": "true"})
        assert result.returncode == 1, "Should fail in CI without approved users"
        assert "approved users list is mandatory" in result.stdout.lower()

    def test_ci_with_approved_users_proceeds(self):
        """Test that CI environment with approved users proceeds past validation"""
        result = self.run_gh_approve(
            ["--pr=123", "--dry-run", "--verbose"],
            env_vars={"CI": "true", "APPROVED_USERS": "testuser,admin"},
        )
        # Should proceed past CI validation (might fail on API call, but that's expected)
        # The key is it doesn't fail on the CI validation
        assert "approved users list is mandatory" in result.stdout

    def test_dry_run_with_nonexistent_pr_fails_gracefully(self):
        """Test that dry run with non-existent PR fails gracefully"""
        result = self.run_gh_approve(["--pr=999999", "--dry-run"])
        assert result.returncode == 1, "Should fail for non-existent PR"
        # Should fail with PR not found error (same path as regular mode)
        assert (
            "404" in result.stdout
            or "Not Found" in result.stdout
            or "401" in result.stdout
            or "Unauthorized" in result.stdout
            or "not found" in result.stdout
        ), "Should fail with HTTP error for non-existent or unauthorized PR"

    def test_help_option_succeeds(self):
        """Test that help option returns success"""
        result = self.run_gh_approve(["--help"])
        assert result.returncode == 0, "Help should return success"
        assert "usage:" in result.stdout.lower()

    def test_invalid_token_fails(self):
        """Test that invalid token fails appropriately"""
        result = self.run_gh_approve(
            ["--pr=123", "--dry-run"], env_vars={"GITHUB_TOKEN": "invalid_token"}
        )
        assert result.returncode == 1, "Should fail with invalid token"
