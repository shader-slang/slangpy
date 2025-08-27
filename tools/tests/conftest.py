# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pytest configuration and fixtures for GitHub tools tests"""

import os
import pytest
from typing import Dict, Optional


@pytest.fixture(autouse=True)
def isolate_environment():
    """
    Automatically isolate environment variables for each test.

    This fixture runs automatically for every test, capturing the current
    environment state before the test and restoring it afterwards.
    """
    # List of environment variables that tests might modify
    env_vars_to_isolate = [
        "CI",
        "APPROVED_USERS",
        "GITHUB_TOKEN",
        "GITHUB_REPOSITORY",
        "GITHUB_REPOSITORY_OWNER",
        "GITHUB_REPOSITORY_NAME",
        "GITHUB_REF",
        "GITHUB_PR_NUMBER",
        "GITHUB_EVENT_NUMBER",
        "GITHUB_EVENT_NAME",
    ]

    # Capture current environment state
    original_env: Dict[str, Optional[str]] = {}
    for var in env_vars_to_isolate:
        original_env[var] = os.environ.get(var)

    # Yield control to the test
    yield

    # Restore original environment state
    for var, original_value in original_env.items():
        if original_value is not None:
            os.environ[var] = original_value
        else:
            # Remove the variable if it wasn't originally set
            os.environ.pop(var, None)


@pytest.fixture
def clean_environment():
    """
    Provide a clean environment with all relevant variables removed.

    Use this fixture when you want to start with a completely clean slate.
    """
    env_vars_to_clean = [
        "CI",
        "APPROVED_USERS",
        "GITHUB_TOKEN",
        "GITHUB_REPOSITORY",
        "GITHUB_REPOSITORY_OWNER",
        "GITHUB_REPOSITORY_NAME",
        "GITHUB_REF",
        "GITHUB_PR_NUMBER",
        "GITHUB_EVENT_NUMBER",
        "GITHUB_EVENT_NAME",
    ]

    # Remove all relevant environment variables
    for var in env_vars_to_clean:
        if var in os.environ:
            del os.environ[var]

    yield


@pytest.fixture
def ci_environment():
    """
    Provide a CI environment setup.

    Sets CI=true and provides a clean environment for CI testing.
    """
    # Clean environment first
    env_vars_to_clean = [
        "CI",
        "APPROVED_USERS",
        "GITHUB_TOKEN",
        "GITHUB_REPOSITORY",
        "GITHUB_REPOSITORY_OWNER",
        "GITHUB_REPOSITORY_NAME",
        "GITHUB_REF",
        "GITHUB_PR_NUMBER",
        "GITHUB_EVENT_NUMBER",
        "GITHUB_EVENT_NAME",
    ]

    for var in env_vars_to_clean:
        if var in os.environ:
            del os.environ[var]

    # Set up CI environment
    os.environ["CI"] = "true"

    yield


@pytest.fixture
def fake_github_token():
    """
    Provide a fake GitHub token for testing.

    This prevents tests from failing due to missing GITHUB_TOKEN
    while still allowing testing of token-related functionality.
    """
    os.environ["GITHUB_TOKEN"] = "fake_token_for_testing"
    yield "fake_token_for_testing"
