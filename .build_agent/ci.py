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


# struggling to get sgl to install via requirements - install directly here instead
run_command("pip install git+https://gitlab-master.nvidia.com/skallweit/sgl.git")

# install requirements for building
run_command("pip install -r .build_agent/requirements.txt")

# run precommit
run_command("pre-commit run --all-files")

# run tests
run_command("pytest tests")
