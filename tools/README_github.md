# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# GitHub Tools

This directory contains tools for interacting with GitHub repositories and pull requests.

## Scripts

### `gh_approve.py`

A tool for approving GitHub pull requests from the command line or CI environments.

#### Features

- Approve pull requests with optional comments
- Support for both command-line arguments and environment variables
- Dry-run mode for testing
- Automatic detection of GitHub Actions environment variables
- Detailed error handling and status reporting

#### Usage

```bash
# Basic usage with environment token
export GITHUB_TOKEN=your_token_here
python tools/gh_approve.py --pr=123

# Usage with token as argument
python tools/gh_approve.py --pr=123 --token=your_token_here

# Add a comment with approval
python tools/gh_approve.py --pr=123 --comment="LGTM! Great work."

# Dry run (test without actually approving)
python tools/gh_approve.py --pr=123 --dry-run

# Different repository
python tools/gh_approve.py --pr=123 --repo-owner=myorg --repo-name=myrepo

# Require approval trigger for security
python tools/gh_approve.py --pr=123 --require-trigger

# Custom trigger pattern
python tools/gh_approve.py --pr=123 --require-trigger --trigger-pattern="\[approve:([^\]]*)\]"
```

#### Command Line Arguments

- `--pr`: Pull request number (optional - auto-detected in CI if not provided)
- `--token`: GitHub personal access token (optional if GITHUB_TOKEN env var is set)
- `--comment`: Optional comment to add with approval
- `--repo-owner`: Repository owner (default: from env vars or "shader-slang")
- `--repo-name`: Repository name (default: from env vars or "slangpy")
- `--dry-run`: Test mode - show what would be done without actually approving
- `--require-trigger`: Require approval trigger in PR description or comments
- `--trigger-pattern`: Custom regex pattern for approval trigger
- `--approved-users`: Comma-separated list of GitHub usernames allowed to use auto-approval

#### Approval Triggers (Security Feature)

For security, you can require specific text in the PR description or comments before allowing approval.
If no trigger is found, the script gracefully skips approval (no error) and provides informational messages.

```bash
# Require default trigger pattern: [auto-approve: comment]
python tools/gh_approve.py --pr=123 --require-trigger

# Custom trigger pattern
python tools/gh_approve.py --pr=123 --require-trigger --trigger-pattern="\[approve:([^\]]*)\]"
```

**Behavior:**
- ✅ **Trigger found**: Proceeds with approval using trigger comment
- ℹ️ **No trigger**: Skips approval gracefully (exit code 0, informational messages)
- ❌ **Error**: Only for actual errors (network, permissions, etc.)

**Example triggers in PR description or comments:**
- `[auto-approve: Ready to merge]`
- `[auto-approve: Dependency update]`
- `[auto-approve: Minor fix]`

#### User Whitelist (Security Feature)

For additional security, you can restrict which GitHub users can use auto-approval by providing a comma-separated list of approved usernames.

```bash
# Restrict to specific users via command line
python tools/gh_approve.py --pr=123 --approved-users="user1,user2,admin"

# Restrict via environment variable
export APPROVED_USERS="user1,user2,admin"
python tools/gh_approve.py --pr=123

# Combine with trigger requirement for layered security
python tools/gh_approve.py --pr=123 --require-trigger --approved-users="trusted-user1,trusted-user2"
```

**Behavior:**
- ✅ **User approved**: PR author is in approved users list → proceeds with approval
- ℹ️ **User not approved**: PR author not in list → skips approval gracefully (exit code 0)
- ✅ **No restriction**: No approved users list set → allows all users (default behavior)
- ⚠️ **CI Environment**: When running in CI (CI=true), approved users list is **mandatory**

**CI Environment Requirements:**
When the script detects it's running in a CI environment (CI=true), the approved users list becomes mandatory for security. This prevents unauthorized auto-approvals in automated environments.
- ✅ **No restriction**: No approved users list set → allows all users (default behavior)

**Features:**
- Case-insensitive username matching
- Handles whitespace in comma-separated lists
- Command line parameter overrides environment variable
- No restriction if neither is set (backward compatible)

This prevents accidental auto-approvals and ensures explicit intent.

#### Environment Variables

The script automatically detects and uses these environment variables:

- `GITHUB_TOKEN`: GitHub personal access token
- `GITHUB_REPOSITORY`: Repository in format "owner/repo" (used in GitHub Actions)
- `GITHUB_REPOSITORY_OWNER`: Repository owner
- `GITHUB_REPOSITORY_NAME`: Repository name
- `APPROVED_USERS`: Comma-separated list of GitHub usernames allowed to use auto-approval (optional)

#### PR Auto-Detection in CI

When running in GitHub Actions, the script can automatically detect the PR number from:

- `GITHUB_REF`: Extract from refs/pull/{number}/merge format
- `GITHUB_PR_NUMBER`: Direct PR number (if available)
- `GITHUB_EVENT_NAME` + `GITHUB_EVENT_NUMBER`: For pull_request events

#### GitHub Actions Integration

In GitHub Actions, the script can run without specifying the PR number:

```yaml
# Simple usage - PR number auto-detected
- name: Approve PR
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: python tools/gh_approve.py --comment="Auto-approved by CI"

# With approval trigger requirement for security
- name: Approve PR (with trigger)
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: python tools/gh_approve.py --require-trigger

# Explicit PR number (still works)
- name: Approve PR
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: python tools/gh_approve.py --pr=${{ github.event.number }}
```

**Recommended secure workflow:**
```yaml
name: Auto-approve PRs
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  auto-approve:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Auto-approve with trigger and required user list
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        APPROVED_USERS: "trusted-dev1,trusted-dev2,bot-user"  # Required in CI
      run: |
        python tools/gh_approve.py --require-trigger --comment="Auto-approved via workflow"
```

**Secure workflow with user whitelist:**
```yaml
name: Auto-approve PRs (Restricted Users)
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  auto-approve:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Auto-approve with trigger and user restriction
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        APPROVED_USERS: "trusted-dev1,trusted-dev2,bot-user"
      run: |
        python tools/gh_approve.py --require-trigger --comment="Auto-approved for trusted user"
```

### `gh_helpers.py`

A helper library providing a Python interface for GitHub API operations.

#### Classes

- `GitHubAPI`: Main class for GitHub API interactions
  - `get_pull_request(pr_number)`: Get PR details
  - `approve_pull_request(pr_number, comment)`: Approve a PR
  - `list_pull_request_reviews(pr_number)`: List PR reviews

#### Functions

- `get_github_token()`: Get token from GITHUB_TOKEN environment variable
- `get_repo_info()`: Get repository owner/name from environment variables
- `get_pr_number_from_ci()`: Auto-detect PR number from GitHub Actions environment

## Requirements

- Python 3.7+
- `requests` library (included in requirements-dev.txt)

## Testing

Run the test script to verify everything works:

```bash
python tools/test_gh_approve.py
```

## Token Setup

### Personal Access Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` scope
3. Set it as environment variable: `export GITHUB_TOKEN=your_token_here`

### GitHub Actions

The `GITHUB_TOKEN` is automatically available in GitHub Actions workflows.
