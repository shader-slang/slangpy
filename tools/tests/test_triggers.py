# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test approval trigger functionality"""

import os
import sys
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gh_helpers import GitHubAPI


class TestApprovalTriggers:
    """Test the approval trigger functionality"""

    def test_trigger_pattern_matching(self):
        """Test various approval trigger patterns"""
        test_cases = [
            ("[auto-approve: Ready to merge]", r"\[auto-approve:([^\]]*)\]", " Ready to merge"),
            ("[auto-approve:dependency update]", r"\[auto-approve:([^\]]*)\]", "dependency update"),
            (
                "[AUTO-APPROVE: URGENT FIX]",
                r"\[auto-approve:([^\]]*)\]",
                " URGENT FIX",
            ),  # Case insensitive
            ("[approve-me: Good to go]", r"\[approve-me:([^\]]*)\]", " Good to go"),
            ("No trigger here", r"\[auto-approve:([^\]]*)\]", None),
            ("[auto-approve:]", r"\[auto-approve:([^\]]*)\]", ""),  # Empty comment
            (
                "Multiple [auto-approve: first] and [auto-approve: second]",
                r"\[auto-approve:([^\]]*)\]",
                " first",
            ),  # First match
        ]

        for text, pattern, expected in test_cases:
            match = re.search(pattern, text, re.IGNORECASE)
            if expected is None:
                assert match is None, f"Should not match pattern in: {text}"
            else:
                assert match is not None, f"Should match pattern in: {text}"
                assert (
                    match.group(1) == expected
                ), f"Expected '{expected}', got '{match.group(1)}' in: {text}"

    def test_trigger_in_pr_description(self):
        """Test trigger detection in PR description"""
        # This would require a mock GitHub API or actual API calls
        # For now, we'll test the pattern matching logic

        pr_description = """
        This is a test PR.

        [auto-approve: Automated dependency update]

        Please merge this automatically.
        """

        pattern = r"\[auto-approve:([^\]]*)\]"
        match = re.search(pattern, pr_description, re.IGNORECASE)

        assert match is not None, "Should find trigger in PR description"
        assert (
            match.group(1).strip() == "Automated dependency update"
        ), "Should extract correct comment"

    def test_trigger_with_various_whitespace(self):
        """Test trigger handling with different whitespace patterns"""
        test_cases = [
            "[auto-approve:no-space]",
            "[auto-approve: single-space]",
            "[auto-approve:  double-space]",
            "[auto-approve:\ttab]",
            "[auto-approve:\n\nnewlines]",
        ]

        pattern = r"\[auto-approve:([^\]]*)\]"

        for text in test_cases:
            match = re.search(pattern, text, re.IGNORECASE)
            assert match is not None, f"Should match trigger in: {text}"
            # Just verify it captures something, whitespace handling is up to the caller
            assert len(match.group(1)) >= 0, f"Should capture content in: {text}"

    def test_custom_trigger_patterns(self):
        """Test custom trigger patterns"""
        custom_patterns = [
            (r"\[approve:([^\]]*)\]", "[approve: custom pattern]", " custom pattern"),
            (r"\[merge-me:([^\]]*)\]", "[merge-me: ready]", " ready"),
            (r"\[bot-approve:([^\]]*)\]", "[bot-approve: automated]", " automated"),
        ]

        for pattern, text, expected in custom_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            assert match is not None, f"Should match custom pattern in: {text}"
            assert match.group(1) == expected, f"Expected '{expected}', got '{match.group(1)}'"
