# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Quick script to list PRs for testing gh_approve.py
"""

import os
import sys

# Add tools directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gh_helpers import GitHubAPI


def list_prs():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set")
        return

    api = GitHubAPI(token)
    try:
        response = api._make_request(
            "GET", "/repos/shader-slang/slangpy/pulls?state=all&per_page=10"
        )
        prs = response.json()

        print("Recent PRs in shader-slang/slangpy:")
        for pr in prs[:10]:
            print(f"#{pr['number']}: {pr['title']} ({pr['state']})")

        if prs:
            print(
                f"\nYou can test with: python tools/gh_approve.py --pr={prs[0]['number']} --dry-run"
            )
        else:
            print("No PRs found")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    list_prs()
