# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import shutil
import subprocess
import sys
import os
from typing import Optional


def get_os():
    """
    Return the OS name (windows, linux, macos).
    """
    platform = sys.platform
    if platform == "win32":
        return "windows"
    elif platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macos"
    else:
        raise NameError(f"Unsupported OS: {sys.platform}")

# Helper to run a command.


def run_command(command: str, shell: bool = True, env: Optional[dict[str, str]] = None):
    if get_os() == "windows":
        command = command.replace("/", "\\")
    if env != None:
        new_env = os.environ.copy()
        new_env.update(env)
        env = new_env
    print(f'Running "{command}" ...')
    sys.stdout.flush()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
        shell=shell,
        env=env,
    )
    assert process.stdout is not None

    out = ""
    while True:
        nextline = process.stdout.readline()
        if nextline == "" and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()
        out += nextline

    process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f'Error running "{command}"')

    return out


FAILED = False


def dependencies(args: argparse.Namespace):
    # struggling to get sgl to install via requirements - install directly here instead
    run_command("pip install --upgrade nv-sgl")

    # install dev requirements
    run_command("pip install --upgrade -r requirements-dev.txt")


def precommit(args: argparse.Namespace):
    run_command("pre-commit run --all-files")


def install(args: argparse.Namespace):
    # install this package as editable
    run_command("pip install --editable .")


def test(args: argparse.Namespace):
    # run tests with native emulation
    env = {}
    mode = 'nat'
    if args.emulated:
        env['SLANGPY_DISABLE_NATIVE'] = '1'
        mode = 'emu'
    if args.device:
        env['SLANGPY_DEVICE'] = args.device
    run_command(f"pytest --junit-xml=test-{mode}-{args.device}-junit.xml", env=env)


def cleanup(args: argparse.Namespace):
    try:
        run_command("pip uninstall -y nv-sgl")
    except Exception as e:
        print(f"WARNING: Cleanup failed with exception:")
        print(e)


def build(args: argparse.Namespace):
    try:
        if os.path.exists("./dist"):
            shutil.rmtree("./dist", ignore_errors=True)
        run_command("pip install build")
        run_command("python -m build")
    except Exception as e:
        print(f"WARNING: Cleanup failed with exception:")
        print(e)


def main():

    # Command line parsing

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--os", type=str, action="store", help="OS (windows, linux, macos)"
    )
    parser.add_argument("--python", type=str, action="store", help="Python version")
    parser.add_argument("--flags", type=str, action="store", help="Additional flags")

    commands = parser.add_subparsers(
        dest="command", required=True, help="sub-command help"
    )

    commands.add_parser("dependencies", help="install dependencies")
    commands.add_parser("install", help="install local slangpy")
    commands.add_parser("cleanup", help="cleanup dependencies")
    commands.add_parser("precommit", help="run precommit hooks")
    commands.add_parser("build", help="build wheels to ./dist")

    test_parser = commands.add_parser("test", help="run unit tests")
    test_parser.add_argument("--emulated", action="store_true",
                             help="run tests with native emulation")
    test_parser.add_argument("--device", action="store",
                             help="device type (d3d12/vulkan/metal)")

    # Read args
    args = parser.parse_args()
    args = vars(args)

    # Inject defaults worked out internally or from environment variables.
    VARS = [
        ("os", "CI_OS", get_os()),
        ("config", "CI_CONFIG", "Debug"),
        ("python", "CI_PYTHON", "3.9"),
        ("flags", "CI_FLAGS", ""),
    ]
    for var, env_var, default_value in VARS:
        if not var in args or args[var] == None:
            args[var] = os.environ[env_var] if env_var in os.environ else default_value

    # Split flags.
    args["flags"] = args["flags"].split(",") if args["flags"] != "" else []

    # Print all args
    print("CI configuration:")
    print(json.dumps(args, indent=4))

    # Convert back to argparse format.
    args = argparse.Namespace(**args)

    # Call the requested command
    commands = {
        "precommit": precommit,
        "dependencies": dependencies,
        "install": install,
        "test": test,
        "cleanup": cleanup,
        "build": build
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    main()
