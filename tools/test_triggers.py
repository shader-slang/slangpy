# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test script for approval trigger functionality
"""

import os
import sys

# Add tools directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gh_helpers import GitHubAPI


def test_trigger_patterns():
    """Test various approval trigger patterns."""

    print("Testing approval trigger patterns...")

    # Test various patterns
    test_cases = [
        ("[auto-approve: Ready to merge]", r"\[auto-approve:([^\]]*)\]", "Ready to merge"),
        ("[auto-approve:dependency update]", r"\[auto-approve:([^\]]*)\]", "dependency update"),
        (
            "[AUTO-APPROVE: URGENT FIX]",
            r"\[auto-approve:([^\]]*)\]",
            "URGENT FIX",
        ),  # Case insensitive
        ("[approve-me: Good to go]", r"\[approve-me:([^\]]*)\]", "Good to go"),
        ("No trigger here", r"\[auto-approve:([^\]]*)\]", None),
        ("[auto-approve:]", r"\[auto-approve:([^\]]*)\]", ""),  # Empty comment
    ]

    import re

    print("\nPattern matching tests:")
    for text, pattern, expected in test_cases:
        match = re.search(pattern, text, re.IGNORECASE)
        result = match.group(1).strip() if match and match.groups() else None
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: '{text}' -> '{result}' (expected: '{expected}')")


def test_with_real_pr():
    """Test trigger detection with real PR if token is available."""

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("\nSkipping real PR test - GITHUB_TOKEN not set")
        return

    print("\nTesting with real PR #459:")

    try:
        api = GitHubAPI(token)

        # Test default pattern
        trigger_comment = api.check_approval_trigger(459)
        print(f"Default pattern result: {trigger_comment}")

        # Test custom pattern
        custom_pattern = r"\[test-approve:([^\]]*)\]"
        trigger_comment = api.check_approval_trigger(459, custom_pattern)
        print(f"Custom pattern result: {trigger_comment}")

        print("SUCCESS: Real PR trigger test completed")

    except Exception as e:
        print(f"ERROR: Real PR test failed: {e}")


if __name__ == "__main__":
    test_trigger_patterns()
    test_with_real_pr()
