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
        raise NameError(f"Error running command: {command}")
    return result


# struggling to get sgl to install via requirements - install directly here instead
run_command(
    "pip install --upgrade --force-reinstall git+https://gitlab-master.nvidia.com/skallweit/sgl.git"
)

# install this package as editable
run_command("pip install --editable .")

# install dev requirements
run_command("pip install -r requirements-dev.txt")

# run precommit
run_command("pre-commit run --all-files")

# run tests
run_command("pytest --junit-xml=junit-test.xml tests")
