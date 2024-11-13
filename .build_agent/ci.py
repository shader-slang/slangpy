import subprocess
import sys
import os

# Helper to run a command.


def run_command(command: str, shell: bool = True):
    print(command)
    sys.stdout.flush()
    result = subprocess.run(command, shell=shell)
    if result.returncode != 0:
        raise NameError(f"Error running command: {command}")
    return result


FAILED = False

try:
    # struggling to get sgl to install via requirements - install directly here instead
    run_command("pip install nv-sgl")

    # install this package as editable
    run_command("pip install --editable .")

    # install dev requirements
    run_command("pip install -r requirements-dev.txt")

    # run precommit
    run_command("pre-commit run --all-files")

    # run tests with native emulation
    # os.environ["SLANGPY_DISABLE_NATIVE"] = "1"
    # run_command("pytest --junit-xml=junit-test-emu.xml")

    # run tests with native
    # del os.environ["SLANGPY_DISABLE_NATIVE"]
    # run_command("pytest --junit-xml=junit-test.xml")
except Exception as e:
    print(e)
    FAILED = True

run_command("pip uninstall -y nv-sgl")
