# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test CI environment behavior in gh_approve.py"""

import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gh_helpers import check_approved_user, is_running_in_ci


class TestCIEnvironment:
    """Test the CI environment detection and approved users requirement"""

    def test_non_ci_environment_no_approved_users_required(self, clean_environment):
        """Test that non-CI environment doesn't require approved users"""
        result = is_running_in_ci()
        assert result == False, "Should not detect CI when CI env var is not set"

        # Should allow user without approved list in non-CI
        result = check_approved_user("testuser")
        assert result == True, "Should allow user when no approved list in non-CI"

    def test_ci_environment_detection(self, ci_environment):
        """Test that CI environment is detected correctly"""
        result = is_running_in_ci()
        assert result == True, "Should detect CI when CI=true"

    def test_ci_environment_requires_approved_users_list(self, ci_environment):
        """Test that CI environment requires approved users list"""
        with pytest.raises(
            ValueError, match="Approved users list is required when running in CI environment"
        ):
            check_approved_user("testuser")

    def test_ci_environment_with_approved_users_list(self, ci_environment):
        """Test that CI environment works with approved users list"""
        result = check_approved_user("testuser", "testuser,anotheruser")
        assert result == True, "Should work with approved list in CI"

    def test_ci_environment_user_not_in_approved_list(self, ci_environment):
        """Test that CI environment rejects user not in approved list"""
        result = check_approved_user("baduser", "testuser,anotheruser")
        assert result == False, "Should reject user not in approved list in CI"

    def test_ci_environment_with_environment_variable(self, ci_environment):
        """Test that CI environment works with APPROVED_USERS environment variable"""
        os.environ["APPROVED_USERS"] = "envuser,testuser"

        result = check_approved_user("envuser")
        assert result == True, "Should work with environment variable in CI"

    def test_ci_false_not_detected_as_ci(self, clean_environment):
        """Test that CI=false is not detected as CI environment"""
        os.environ["CI"] = "false"

        result = is_running_in_ci()
        assert result == False, "Should not detect CI when CI=false"
