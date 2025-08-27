# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test approval workflow integration"""

import os
import sys
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestApprovalIntegration:
    """Test the approval workflow integration"""

    def test_help_includes_approved_users_parameter(self):
        """Test that help message includes --approved-users parameter"""
        result = subprocess.run(
            [sys.executable, "gh_approve.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )

        assert result.returncode == 0, "Help command should succeed"
        assert (
            "--approved-users" in result.stdout
        ), "--approved-users parameter should be documented"

    def test_help_includes_approved_users_env_var(self):
        """Test that help message includes APPROVED_USERS environment variable"""
        result = subprocess.run(
            [sys.executable, "gh_approve.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )

        assert (
            "APPROVED_USERS" in result.stdout
        ), "APPROVED_USERS environment variable should be documented"

    def test_script_imports_successfully(self):
        """Test that gh_approve.py imports without errors"""
        # This tests that all the new imports work correctly
        try:
            import sys
            import os

            tools_dir = os.path.dirname(os.path.dirname(__file__))
            sys.path.insert(0, tools_dir)

            # Import the main module to check for syntax/import errors
            import gh_approve

            assert hasattr(gh_approve, "main"), "Should have main function"

        except ImportError as e:
            assert False, f"Import error: {e}"
        except SyntaxError as e:
            assert False, f"Syntax error: {e}"

    def test_required_functions_available(self):
        """Test that required functions are available from gh_helpers"""
        try:
            import sys
            import os

            tools_dir = os.path.dirname(os.path.dirname(__file__))
            sys.path.insert(0, tools_dir)

            from gh_helpers import check_approved_user, is_running_in_ci, GitHubAPI

            # Verify functions exist and are callable
            assert callable(check_approved_user), "check_approved_user should be callable"
            assert callable(is_running_in_ci), "is_running_in_ci should be callable"
            assert GitHubAPI is not None, "GitHubAPI class should be available"

        except ImportError as e:
            assert False, f"Import error: {e}"
