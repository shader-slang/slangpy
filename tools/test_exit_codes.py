# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test exit codes for different scenarios
"""

import subprocess
import sys
import os


def test_exit_codes():
    """Test that exit codes are correct for different scenarios."""

    print("Testing exit codes for different approval scenarios...\n")

    # Test cases: (description, command_args, expected_exit_code)
    test_cases = [
        ("Dry run without trigger requirement", ["--pr=459", "--dry-run"], 0),
        (
            "Dry run with trigger requirement (no trigger present)",
            ["--pr=459", "--require-trigger", "--dry-run"],
            0,
        ),
        ("No PR argument and not in CI", [], 1),  # Should fail - no PR number
    ]

    base_cmd = [sys.executable, "tools/gh_approve.py"]

    for description, args, expected_code in test_cases:
        print(f"Test: {description}")
        print(f"Command: {' '.join(base_cmd + args)}")

        try:
            result = subprocess.run(
                base_cmd + args,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )

            actual_code = result.returncode
            status = "PASS" if actual_code == expected_code else "FAIL"

            print(f"Expected exit code: {expected_code}")
            print(f"Actual exit code: {actual_code}")
            print(f"Status: {status}")

            if status == "FAIL":
                print(
                    "STDOUT:",
                    result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout,
                )
                print(
                    "STDERR:",
                    result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr,
                )

        except Exception as e:
            print(f"ERROR: Failed to run test: {e}")

        print("-" * 60)


if __name__ == "__main__":
    test_exit_codes()
