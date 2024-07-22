import argparse
import os
import subprocess
import sys


# Helper to run a command.
def run_command(command, shell=True):
    print(command)
    sys.stdout.flush()
    result = subprocess.run(command, shell=shell)
    if result.returncode != 0:
        raise (f"Error running command: {command}")
    return result


run_command("pip install --editable .")

run_command("pytest tests")
