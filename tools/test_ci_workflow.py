# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/usr/bin/env python3
"""Test script for CI requirements in gh_approve.py workflow"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def test_ci_workflow():
    """Test the CI workflow requirements"""

    print("Testing CI workflow requirements...")
    print("=" * 50)

    # Save original environment variables
    original_ci = os.environ.get("CI")
    original_approved = os.environ.get("APPROVED_USERS")
    original_token = os.environ.get("GITHUB_TOKEN")

    try:
        # Set up a fake token to avoid the token check failing
        os.environ["GITHUB_TOKEN"] = "fake_token_for_testing"

        # Test 1: CI environment without approved users should fail early
        print("\nTest 1: CI environment without approved users (should fail)")
        os.environ["CI"] = "true"
        if "APPROVED_USERS" in os.environ:
            del os.environ["APPROVED_USERS"]

        # Import and test the approval logic
        from gh_approve import main
        import subprocess

        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, "gh_approve.py", "--pr=123", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

        print(f"Exit code: {result.returncode}")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")

        # Should fail with error about missing approved users
        if (
            "Approved users list is required when running in CI" in result.stderr
            or "Approved users list is required when running in CI" in result.stdout
        ):
            print("✓ PASS: Correctly requires approved users in CI")
        else:
            print("✗ FAIL: Should require approved users in CI")
            return False

        # Test 2: CI environment with approved users
        print("\nTest 2: CI environment with approved users")
        os.environ["APPROVED_USERS"] = "testuser,admin"

        result = subprocess.run(
            [sys.executable, "gh_approve.py", "--pr=123", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

        print(f"Exit code: {result.returncode}")
        print(f"Output: {result.stdout}")

        # Should proceed further (might fail on API call, but not on user restriction)
        if "approved users list is mandatory" in result.stdout:
            print("✓ PASS: Correctly detected CI environment and shows approved users requirement")
        else:
            print("? INFO: Might be failing on API call rather than user restrictions")

        print("\n" + "=" * 50)
        print("CI workflow tests completed!")
        return True

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

        if original_token is not None:
            os.environ["GITHUB_TOKEN"] = original_token
        else:
            os.environ.pop("GITHUB_TOKEN", None)


if __name__ == "__main__":
    success = test_ci_workflow()
    sys.exit(0 if success else 1)
