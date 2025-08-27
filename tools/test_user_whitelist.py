# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/usr/bin/env python3
"""Test script for user whitelist functionality in gh_approve.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gh_helpers import check_approved_user


def test_user_whitelist():
    """Test the user whitelist functionality"""

    print("Testing user whitelist functionality...")
    print("=" * 50)

    # Test 1: No approved users set (should allow all)
    print("\nTest 1: No approved users list (should allow all users)")
    result = check_approved_user("testuser")
    print(f"check_approved_user('testuser') = {result}")
    assert result == True, "Should allow all users when no list is set"
    print("✓ PASS")

    # Test 2: User in approved list
    print("\nTest 2: User in approved list")
    result = check_approved_user("testuser", "testuser,anotheruser")
    print(f"check_approved_user('testuser', 'testuser,anotheruser') = {result}")
    assert result == True, "Should allow user in approved list"
    print("✓ PASS")

    # Test 3: User not in approved list
    print("\nTest 3: User not in approved list")
    result = check_approved_user("baduser", "testuser,anotheruser")
    print(f"check_approved_user('baduser', 'testuser,anotheruser') = {result}")
    assert result == False, "Should not allow user not in approved list"
    print("✓ PASS")

    # Test 4: Case insensitive matching
    print("\nTest 4: Case insensitive matching")
    result = check_approved_user("TestUser", "testuser,anotheruser")
    print(f"check_approved_user('TestUser', 'testuser,anotheruser') = {result}")
    assert result == True, "Should be case insensitive"
    print("✓ PASS")

    # Test 5: Whitespace handling
    print("\nTest 5: Whitespace handling")
    result = check_approved_user("testuser", " testuser , anotheruser ")
    print(f"check_approved_user('testuser', ' testuser , anotheruser ') = {result}")
    assert result == True, "Should handle whitespace in approved list"
    print("✓ PASS")

    # Test 6: Empty approved list
    print("\nTest 6: Empty approved list")
    result = check_approved_user("testuser", "")
    print(f"check_approved_user('testuser', '') = {result}")
    assert result == True, "Should allow all users when approved list is empty"
    print("✓ PASS")

    # Test 7: Environment variable (clean test)
    print("\nTest 7: Environment variable")
    old_env = os.environ.get("APPROVED_USERS")
    try:
        os.environ["APPROVED_USERS"] = "envuser,testuser"
        result = check_approved_user("envuser")
        print(f"check_approved_user('envuser') with APPROVED_USERS='envuser,testuser' = {result}")
        assert result == True, "Should use environment variable"
        print("✓ PASS")
    finally:
        if old_env is not None:
            os.environ["APPROVED_USERS"] = old_env
        else:
            os.environ.pop("APPROVED_USERS", None)

    print("\n" + "=" * 50)
    print("All user whitelist tests passed! ✓")


if __name__ == "__main__":
    test_user_whitelist()
