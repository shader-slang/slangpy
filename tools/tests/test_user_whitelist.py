# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test user whitelist functionality in gh_approve.py"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gh_helpers import check_approved_user


class TestUserWhitelist:
    """Test the user whitelist functionality"""

    def test_no_approved_users_allows_all(self, clean_environment):
        """Test that no approved users list allows all users"""
        result = check_approved_user("testuser")
        assert result == True, "Should allow all users when no list is set"

    def test_user_in_approved_list(self, clean_environment):
        """Test that user in approved list is allowed"""
        result = check_approved_user("testuser", "testuser,anotheruser")
        assert result == True, "Should allow user in approved list"

    def test_user_not_in_approved_list(self, clean_environment):
        """Test that user not in approved list is rejected"""
        result = check_approved_user("baduser", "testuser,anotheruser")
        assert result == False, "Should not allow user not in approved list"

    def test_case_insensitive_matching(self, clean_environment):
        """Test that username matching is case insensitive"""
        result = check_approved_user("TestUser", "testuser,anotheruser")
        assert result == True, "Should be case insensitive"

    def test_whitespace_handling(self, clean_environment):
        """Test that whitespace in approved list is handled correctly"""
        result = check_approved_user("testuser", " testuser , anotheruser ")
        assert result == True, "Should handle whitespace in approved list"

    def test_empty_approved_list_allows_all(self, clean_environment):
        """Test that empty approved list allows all users"""
        result = check_approved_user("testuser", "")
        assert result == True, "Should allow all users when approved list is empty"

    def test_environment_variable(self, clean_environment):
        """Test that environment variable is used when no direct list provided"""
        os.environ["APPROVED_USERS"] = "envuser,testuser"
        result = check_approved_user("envuser")
        assert result == True, "Should use environment variable"
