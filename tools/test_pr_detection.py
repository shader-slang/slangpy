# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test script for PR auto-detection functionality
"""

import os
import sys

# Add tools directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gh_helpers import get_pr_number_from_ci


def test_pr_detection():
    """Test PR number detection from various CI environment variables."""

    print("Testing PR auto-detection from CI environment variables...")

    # Save current environment
    original_env = {}
    test_vars = ["GITHUB_PR_NUMBER", "GITHUB_REF", "GITHUB_EVENT_NAME", "GITHUB_EVENT_NUMBER"]
    for var in test_vars:
        original_env[var] = os.environ.get(var)
        # Clear the variable
        if var in os.environ:
            del os.environ[var]

    try:
        # Test 1: No CI environment
        print("\n1. Testing with no CI environment:")
        pr_num = get_pr_number_from_ci()
        print(f"   Result: {pr_num} (expected: None)")

        # Test 2: GITHUB_PR_NUMBER
        print("\n2. Testing with GITHUB_PR_NUMBER:")
        os.environ["GITHUB_PR_NUMBER"] = "123"
        pr_num = get_pr_number_from_ci()
        print(f"   Result: {pr_num} (expected: 123)")
        del os.environ["GITHUB_PR_NUMBER"]

        # Test 3: GITHUB_REF (pull request format)
        print("\n3. Testing with GITHUB_REF:")
        os.environ["GITHUB_REF"] = "refs/pull/456/merge"
        pr_num = get_pr_number_from_ci()
        print(f"   Result: {pr_num} (expected: 456)")
        del os.environ["GITHUB_REF"]

        # Test 4: GITHUB_EVENT_NAME + GITHUB_EVENT_NUMBER
        print("\n4. Testing with GITHUB_EVENT_NAME + GITHUB_EVENT_NUMBER:")
        os.environ["GITHUB_EVENT_NAME"] = "pull_request"
        os.environ["GITHUB_EVENT_NUMBER"] = "789"
        pr_num = get_pr_number_from_ci()
        print(f"   Result: {pr_num} (expected: 789)")

        print("\nSUCCESS: All PR detection tests completed")

    finally:
        # Restore original environment
        for var, value in original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]


def test_manual_ci_simulation():
    """Simulate a GitHub Actions environment and test the full script."""

    print("\n" + "=" * 60)
    print("Simulating GitHub Actions environment for PR #459:")
    print("=" * 60)

    # Simulate GitHub Actions environment for PR #459
    os.environ["GITHUB_REF"] = "refs/pull/459/merge"
    os.environ["GITHUB_EVENT_NAME"] = "pull_request"
    os.environ["GITHUB_REPOSITORY"] = "shader-slang/slangpy"

    try:
        pr_num = get_pr_number_from_ci()
        print(f"Auto-detected PR number: {pr_num}")

        if pr_num == 459:
            print("SUCCESS: Correctly detected PR #459 from simulated CI environment")
        else:
            print(f"ERROR: Expected 459, got {pr_num}")

    finally:
        # Clean up
        for var in ["GITHUB_REF", "GITHUB_EVENT_NAME"]:
            if var in os.environ:
                del os.environ[var]


if __name__ == "__main__":
    test_pr_detection()
    test_manual_ci_simulation()
