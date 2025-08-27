# GitHub Tools Tests

This directory contains pytest tests for the GitHub approval tools.

## Running Tests

To run all tests:
```bash
pytest tools/tests
```

To run tests with verbose output:
```bash
pytest tools/tests -v
```

To run a specific test file:
```bash
pytest tools/tests/test_user_whitelist.py
```

To run a specific test method:
```bash
pytest tools/tests/test_user_whitelist.py::TestUserWhitelist::test_case_insensitive_matching
```

## Test Files

- `test_user_whitelist.py` - Tests for user whitelist functionality
- `test_ci_environment.py` - Tests for CI environment detection and requirements
- `test_triggers.py` - Tests for approval trigger pattern matching
- `test_exit_codes.py` - Tests for exit code behavior in different scenarios
- `test_api_integration.py` - Tests for GitHub API integration functionality
- `test_approval_integration.py` - Tests for overall approval workflow integration

## Test Environment

Tests automatically handle environment variable cleanup and restoration to avoid interference between tests. Each test class uses `setup_method()` and `teardown_method()` to ensure a clean testing environment.

## Coverage

The tests cover:
- ✅ User whitelist functionality (case insensitive, environment variables)
- ✅ CI environment detection and mandatory user requirements
- ✅ Approval trigger pattern matching
- ✅ Exit code behavior for various scenarios
- ✅ GitHub API integration basics
- ✅ Command line argument processing
- ✅ Error handling and validation
