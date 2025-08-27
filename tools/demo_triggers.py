# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Demo script showing approval trigger functionality
"""

import re


def demo_trigger_patterns():
    """Demonstrate how approval triggers work."""

    print("=== GitHub PR Approval Trigger Demo ===\n")

    # Sample PR descriptions and comments
    samples = [
        {
            "type": "PR Description",
            "text": """
# Fix memory leak in renderer

This PR fixes a memory leak that was causing performance issues.

[auto-approve: Minor bug fix, ready to merge]

## Changes
- Fixed malloc/free mismatch
- Added unit tests
            """,
            "expected": "Minor bug fix, ready to merge",
        },
        {
            "type": "PR Comment",
            "text": "[auto-approve: Dependency update looks good]",
            "expected": "Dependency update looks good",
        },
        {
            "type": "Case Insensitive",
            "text": "[AUTO-APPROVE: URGENT SECURITY FIX]",
            "expected": "URGENT SECURITY FIX",
        },
        {"type": "No Trigger", "text": "This PR looks good but has no trigger", "expected": None},
        {"type": "Empty Comment", "text": "[auto-approve:]", "expected": ""},
    ]

    pattern = r"\[auto-approve:([^\]]*)\]"

    for sample in samples:
        print(f"Sample: {sample['type']}")
        print(f"Text: {sample['text'][:100]}{'...' if len(sample['text']) > 100 else ''}")

        match = re.search(pattern, sample["text"], re.IGNORECASE)
        result = match.group(1).strip() if match and match.groups() else None

        print(f"Trigger Found: {result}")
        print(f"Expected: {sample['expected']}")
        print(f"Status: {'✓ PASS' if result == sample['expected'] else '✗ FAIL'}")
        print("-" * 50)

    print("\n=== Usage Examples ===")
    print("1. Basic approval with trigger:")
    print("   python tools/gh_approve.py --pr=123 --require-trigger")

    print("\n2. Custom trigger pattern:")
    print(
        '   python tools/gh_approve.py --pr=123 --require-trigger --trigger-pattern="\\[approve:([^\\]]*)\\]"'
    )

    print("\n3. In GitHub Actions:")
    print("   python tools/gh_approve.py --require-trigger")

    print("\n=== Security Benefits ===")
    print("- Prevents accidental auto-approvals")
    print("- Requires explicit intent from PR author or reviewers")
    print("- Customizable trigger patterns for different workflows")
    print("- Works with PR descriptions and comments")


if __name__ == "__main__":
    demo_trigger_patterns()
