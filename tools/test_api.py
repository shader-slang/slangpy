# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Direct API test for approval
"""

import os
import sys
import requests

# Add tools directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gh_helpers import GitHubAPI


def test_approval_api():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set")
        return

    api = GitHubAPI(token)

    # Test the exact API call
    url = f"{api.base_url}/repos/{api.repo_owner}/{api.repo_name}/pulls/459/reviews"
    data = {"event": "APPROVE", "body": "Test approval from API debugging"}

    print(f"Making request to: {url}")
    print(f"Data: {data}")
    print(f"Headers: {api.headers}")

    try:
        response = requests.post(url, headers=api.headers, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 422:
            print("422 Error - checking response details...")
            if response.headers.get("content-type", "").startswith("application/json"):
                error_details = response.json()
                print(f"Error details: {error_details}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_approval_api()
