# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Example usage and test script for gh_approve.py
"""

import os
import subprocess
import sys


def test_gh_approve_help():
    """Test that gh_approve.py shows help correctly."""
    try:
        result = subprocess.run(
            [sys.executable, "tools/gh_approve.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        print("SUCCESS: Help command executed successfully")
        print("Return code:", result.returncode)
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: Error running help command: {e}")
        return False


def test_gh_approve_no_args():
    """Test that gh_approve.py fails appropriately when no PR is specified."""
    try:
        result = subprocess.run(
            [sys.executable, "tools/gh_approve.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        print("SUCCESS: No args test executed")
        print("Return code:", result.returncode)
        print("Error output contains required argument:", "--pr" in result.stderr)
        return result.returncode != 0 and "--pr" in result.stderr
    except Exception as e:
        print(f"ERROR: Error running no args test: {e}")
        return False


def show_usage_examples():
    """Show usage examples for the script."""
    print("\n" + "=" * 60)
    print("GitHub PR Approval Tool Usage Examples")
    print("=" * 60)

    print("\n1. Basic usage with token from environment:")
    print("   export GITHUB_TOKEN=your_token_here")
    print("   python tools/gh_approve.py --pr=123")

    print("\n2. Usage with token as argument:")
    print("   python tools/gh_approve.py --pr=123 --token=your_token_here")

    print("\n3. Usage with custom comment:")
    print("   python tools/gh_approve.py --pr=123 --comment='LGTM! Great work.'")

    print("\n4. Dry run to test without approving:")
    print("   python tools/gh_approve.py --pr=123 --dry-run")

    print("\n5. Usage for different repository:")
    print("   python tools/gh_approve.py --pr=123 --repo-owner=myorg --repo-name=myrepo")

    print("\n6. CI/GitHub Actions usage:")
    print("   Environment variables are automatically detected:")
    print("   - GITHUB_TOKEN")
    print("   - GITHUB_REPOSITORY (format: owner/repo)")
    print("   - GITHUB_REPOSITORY_OWNER")
    print("   - GITHUB_REPOSITORY_NAME")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Testing gh_approve.py script...")

    test1_passed = test_gh_approve_help()
    test2_passed = test_gh_approve_no_args()

    print(f"\nTest Results:")
    print(f"Help command test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"No args test: {'PASSED' if test2_passed else 'FAILED'}")

    show_usage_examples()

    if test1_passed and test2_passed:
        print("\nSUCCESS: All tests passed! The gh_approve.py script is ready to use.")
    else:
        print("\nERROR: Some tests failed. Please check the script.")
        sys.exit(1)
