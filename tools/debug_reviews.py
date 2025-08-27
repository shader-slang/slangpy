# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Debug script to check PR reviews
"""

import os
import sys

# Add tools directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gh_helpers import GitHubAPI


def check_reviews():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set")
        return

    api = GitHubAPI(token)
    try:
        reviews = api.list_pull_request_reviews(459)
        print(f"Existing reviews on PR #459: {len(reviews)}")
        for review in reviews:
            print(f"- Review by {review['user']['login']}: {review['state']} (ID: {review['id']})")

        # Also check PR details
        pr = api.get_pull_request(459)
        print(f"\nPR Details:")
        print(f"- State: {pr['state']}")
        print(f"- Draft: {pr.get('draft', False)}")
        print(f"- Mergeable: {pr.get('mergeable', 'unknown')}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_reviews()
