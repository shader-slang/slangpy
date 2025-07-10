# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Script for running CI tasks.
"""

import os
import sys
import platform
import argparse
import subprocess
import json
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent


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


def get_platform():
    """
    Return the platform name (x86_64, aarch64).
    """
    machine = platform.machine()
    if machine == "x86_64" or machine == "AMD64":
        return "x86_64"
    elif machine == "aarch64" or machine == "arm64":
        return "aarch64"
    else:
        raise NameError(f"Unsupported platform: {machine}")


def get_default_compiler():
    """
    Return the default compiler name for the current OS (msvc, gcc, clang).
    """
    if get_os() == "windows":
        return "msvc"
    elif get_os() == "linux":
        return "gcc"
    elif get_os() == "macos":
        return "clang"
    else:
        raise NameError(f"Unsupported OS: {get_os()}")


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


def get_changed_env(command: str):
    if get_os() == "windows":
        command = command.replace("/", "\\") + " && set"
    else:
        command = command + " && env"
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        shell=True,
        executable="/bin/bash" if get_os() != "windows" else None,
    )
    if result.returncode != 0:
        raise NameError(f'Error running "{command}"')
    env_vars = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in ["_"]:
                continue
            curr = os.getenv(key)
            if curr is None or curr != value:
                env_vars[key] = value
    return env_vars


def get_python_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_DIR)
    return env


def setup(args: Any):
    if args.os == "windows":
        run_command("./setup.bat")
    else:
        run_command("./setup.sh")


def configure(args: Any):
    cmd = f"{args.cmake} --preset {args.preset}"
    if "header-validation" in args.flags:
        cmd += " -DSGL_ENABLE_HEADER_VALIDATION=ON"
    if "coverage" in args.flags:
        cmd += " -DSGL_ENABLE_COVERAGE=ON"
    if args.cmake_args != "":
        cmd += " " + args.cmake_args
    run_command(cmd)


def build(args: Any):
    run_command(f"{args.cmake} --build build/{args.preset} --config {args.config}")


def unit_test_cpp(args: Any):
    out = run_command(f"{args.bin_dir}/sgl_tests -r=console,junit")
    # doctest outputs both regular output and junit xml report on stdout
    # filter out regular output and write remaining to junit xml file
    report = "\n".join(filter(lambda line: line.strip().startswith("<"), out.splitlines()))
    os.makedirs("reports", exist_ok=True)
    with open("reports/doctest-junit.xml", "w") as f:
        f.write(report)


def typing_check_python(args: Any):
    env = get_python_env()
    run_command(f"pyright", env=env)


def unit_test_python(args: Any):
    env = get_python_env()
    os.makedirs("reports", exist_ok=True)
    cmd = "pytest slangpy/tests -ra --junit-xml=reports/pytest-junit.xml"
    if args.parallel:
        cmd += " -n auto --maxprocesses=8"
    run_command(cmd, env=env)


def test_examples(args: Any):
    env = get_python_env()
    cmd = "pytest samples/tests -vra"
    if args.parallel:
        cmd += " -n auto --maxprocesses=8"
    run_command(cmd, env=env)


def coverage_report(args: Any):
    if not "coverage" in args.flags:
        print("Coverage flag not set, skipping coverage report.")
    os.makedirs("reports", exist_ok=True)
    run_command(f"gcovr -r . -f src/sgl --html reports/coverage.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os", type=str, action="store", help="OS (windows, linux, macos)")
    parser.add_argument("--platform", type=str, action="store", help="Platform (x86_64, aarch64)")
    parser.add_argument("--compiler", type=str, action="store", help="Compiler (msvc, gcc, clang)")
    parser.add_argument("--config", type=str, action="store", help="Config (Release, Debug)")
    parser.add_argument("--python", type=str, action="store", help="Python version")
    parser.add_argument("--flags", type=str, action="store", help="Additional flags")
    parser.add_argument("--cmake-args", type=str, action="store", help="Additional CMake arguments")

    commands = parser.add_subparsers(dest="command", required=True, help="sub-command help")

    parser_setup = commands.add_parser("setup", help="run setup.bat or setup.sh")

    parser_configure = commands.add_parser("configure", help="run cmake configure")

    parser_build = commands.add_parser("build", help="run cmake build")

    parser_test_cpp = commands.add_parser("unit-test-cpp", help="run unit tests (c++)")

    parser_typing_check_python = commands.add_parser(
        "typing-check-python", help="run pyright typing checks (python)"
    )

    parser_test_python = commands.add_parser("unit-test-python", help="run unit tests (python)")
    parser_test_python.add_argument(
        "-p", "--parallel", action="store_true", help="run tests in parallel"
    )

    parser_test_examples = commands.add_parser("test-examples", help="run examples tests")
    parser_test_examples.add_argument(
        "-p", "--parallel", action="store_true", help="run tests in parallel"
    )

    parser_coverage_report = commands.add_parser("coverage-report", help="generate coverage report")

    args = parser.parse_args()
    args = vars(args)

    VARS = [
        ("os", "CI_OS", get_os()),
        ("platform", "CI_PLATFORM", get_platform()),
        ("compiler", "CI_COMPILER", get_default_compiler()),
        ("config", "CI_CONFIG", "Debug"),
        ("python", "CI_PYTHON", "3.9"),
        ("flags", "CI_FLAGS", ""),
        ("cmake_args", "CI_CMAKE_ARGS", ""),
    ]

    for var, env_var, default_value in VARS:
        if not var in args or args[var] == None:
            args[var] = os.environ[env_var] if env_var in os.environ else default_value

    # Split flags.
    args["flags"] = args["flags"].split(",") if args["flags"] != "" else []

    # Determine cmake executable path.
    args["cmake"] = {
        "windows": "cmake.exe",
        "linux": "cmake",
        "macos": "cmake",
    }[args["os"]]

    # Determine cmake preset.
    preset = args["os"] + "-" + args["compiler"]
    if args["os"] == "macos":
        if args["platform"] == "x86_64":
            preset = preset.replace("macos", "macos-x64")
        elif args["platform"] == "aarch64":
            preset = preset.replace("macos", "macos-arm64")
    args["preset"] = preset

    # Determine binary directory.
    bin_dir = f"./build/{args['preset']}/{args['config']}"
    args["bin_dir"] = bin_dir

    print("CI configuration:")
    print(json.dumps(args, indent=4))

    args = argparse.Namespace(**args)

    {
        "setup": setup,
        "configure": configure,
        "build": build,
        "unit-test-cpp": unit_test_cpp,
        "typing-check-python": typing_check_python,
        "unit-test-python": unit_test_python,
        "test-examples": test_examples,
        "coverage-report": coverage_report,
    }[args.command](args)

    return 0


if __name__ == "__main__":
    main()
