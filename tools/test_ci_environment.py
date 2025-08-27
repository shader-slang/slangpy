# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/usr/bin/env python3
"""Test script for CI environment behavior in gh_approve.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gh_helpers import check_approved_user, is_running_in_ci


def test_ci_environment():
    """Test the CI environment detection and approved users requirement"""

    print("Testing CI environment behavior...")
    print("=" * 50)

    # Save original environment variables
    original_ci = os.environ.get("CI")
    original_approved = os.environ.get("APPROVED_USERS")

    try:
        # Clean slate - remove all relevant env vars
        for var in ["CI", "APPROVED_USERS"]:
            if var in os.environ:
                del os.environ[var]
        # Test 1: Non-CI environment (should work as before)
        print("\nTest 1: Non-CI environment (no approved users required)")
        if "CI" in os.environ:
            del os.environ["CI"]

        result = is_running_in_ci()
        print(f"is_running_in_ci() = {result}")
        assert result == False, "Should not detect CI when CI env var is not set"

        # Should allow user without approved list in non-CI
        result = check_approved_user("testuser")
        print(f"check_approved_user('testuser') = {result}")
        assert result == True, "Should allow user when no approved list in non-CI"
        print("✓ PASS")

        # Test 2: CI environment detection
        print("\nTest 2: CI environment detection")
        os.environ["CI"] = "true"

        result = is_running_in_ci()
        print(f"is_running_in_ci() with CI=true = {result}")
        assert result == True, "Should detect CI when CI=true"
        print("✓ PASS")

        # Test 3: CI environment requires approved users list
        print("\nTest 3: CI environment requires approved users list")
        try:
            result = check_approved_user("testuser")
            print(
                f"check_approved_user('testuser') in CI without approved list should raise ValueError"
            )
            assert False, "Should raise ValueError when no approved list in CI"
        except ValueError as e:
            print(f"✓ PASS: Correctly raised ValueError: {e}")

        # Test 4: CI environment with approved users list
        print("\nTest 4: CI environment with approved users list")
        result = check_approved_user("testuser", "testuser,anotheruser")
        print(f"check_approved_user('testuser', 'testuser,anotheruser') in CI = {result}")
        assert result == True, "Should work with approved list in CI"
        print("✓ PASS")

        # Test 5: CI environment with user not in approved list
        print("\nTest 5: CI environment with user not in approved list")
        result = check_approved_user("baduser", "testuser,anotheruser")
        print(f"check_approved_user('baduser', 'testuser,anotheruser') in CI = {result}")
        assert result == False, "Should reject user not in approved list in CI"
        print("✓ PASS")

        # Test 6: CI environment with environment variable
        print("\nTest 6: CI environment with APPROVED_USERS environment variable")
        old_approved = os.environ.get("APPROVED_USERS")
        try:
            os.environ["APPROVED_USERS"] = "envuser,testuser"
            result = check_approved_user("envuser")
            print(
                f"check_approved_user('envuser') with APPROVED_USERS='envuser,testuser' in CI = {result}"
            )
            assert result == True, "Should work with environment variable in CI"
            print("✓ PASS")
        finally:
            if old_approved is not None:
                os.environ["APPROVED_USERS"] = old_approved
            else:
                os.environ.pop("APPROVED_USERS", None)

        # Test 7: CI=false should not be detected as CI
        print("\nTest 7: CI=false should not be detected as CI")
        os.environ["CI"] = "false"
        result = is_running_in_ci()
        print(f"is_running_in_ci() with CI=false = {result}")
        assert result == False, "Should not detect CI when CI=false"
        print("✓ PASS")

    finally:
        # Restore original environment variables
        if original_ci is not None:
            os.environ["CI"] = original_ci
        else:
            os.environ.pop("CI", None)

        if original_approved is not None:
            os.environ["APPROVED_USERS"] = original_approved
        else:
            os.environ.pop("APPROVED_USERS", None)

    print("\n" + "=" * 50)
    print("All CI environment tests passed! ✓")


if __name__ == "__main__":
    test_ci_environment()
