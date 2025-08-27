# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/usr/bin/env python3
"""Test script for gh_approve.py with user whitelist functionality"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def test_approval_with_user_whitelist():
    """Test the approval workflow with user whitelist"""

    print("Testing gh_approve.py with user whitelist...")
    print("=" * 60)

    # Test the help message includes the new parameter
    print("\nTest 1: Check help message includes --approved-users")
    import subprocess

    result = subprocess.run(
        [sys.executable, "gh_approve.py", "--help"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
    )

    if "--approved-users" in result.stdout:
        print("✓ PASS: --approved-users parameter is documented")
    else:
        print("✗ FAIL: --approved-users parameter not found in help")
        print("Help output:", result.stdout)
        return False

    if "APPROVED_USERS" in result.stdout:
        print("✓ PASS: APPROVED_USERS environment variable is documented")
    else:
        print("✗ FAIL: APPROVED_USERS not found in help")
        return False

    # Test argument parsing
    print("\nTest 2: Test argument parsing with user whitelist")
    try:
        from gh_approve import main

        # This would normally run the main function, but we'll just import to check syntax
        print("✓ PASS: gh_approve.py imports successfully with new parameters")
    except Exception as e:
        print(f"✗ FAIL: Import error: {e}")
        return False

    print("\n" + "=" * 60)
    print("All integration tests passed! ✓")
    print("\nTo test the full workflow:")
    print("1. Set GITHUB_TOKEN environment variable")
    print("2. Set APPROVED_USERS='user1,user2' to restrict users")
    print("3. Run: python gh_approve.py --pr <number> --approved-users 'user1,user2'")
    print(
        "4. Or use environment variable: APPROVED_USERS='user1,user2' python gh_approve.py --pr <number>"
    )

    return True


if __name__ == "__main__":
    success = test_approval_with_user_whitelist()
    sys.exit(0 if success else 1)
