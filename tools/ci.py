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
from typing import Any, Optional, Union

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


def run_command(
    command: Union[str, list[str]], shell: bool = True, env: Optional[dict[str, str]] = None
):
    if isinstance(command, str):
        command = [command]
    if get_os() == "windows":
        command[0] = command[0].replace("/", "\\")
    if env != None:
        new_env = os.environ.copy()
        new_env.update(env)
        env = new_env
    print(f'Running "{" ".join(command)}" ...')
    sys.stdout.flush()

    if shell:
        command = " ".join(command)

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
    cmd = [args.cmake, "--preset", args.preset]
    if "header-validation" in args.flags:
        cmd += ["-DSGL_ENABLE_HEADER_VALIDATION=ON"]
    if "coverage" in args.flags:
        cmd += ["-DSGL_ENABLE_COVERAGE=ON"]
    if args.cmake_args != "":
        cmd += args.cmake_args.split()
    run_command(cmd)


def build(args: Any):
    run_command([args.cmake, "--build", f"build/{args.preset}", "--config", args.config])


def unit_test_cpp(args: Any):
    run_command([f"{args.bin_dir}/sgl_tests"])


def typing_check_python(args: Any):
    env = get_python_env()
    run_command(f"pyright", env=env)


def unit_test_python(args: Any):
    env = get_python_env()
    os.makedirs("reports", exist_ok=True)
    cmd = ["pytest", "slangpy/tests", "-vra"]
    if args.parallel:
        cmd += ["-n", "auto", "--maxprocesses=4"]
    run_command(cmd, env=env)


def test_examples(args: Any):
    env = get_python_env()
    cmd = ["pytest", "samples/tests", "-vra"]
    if args.parallel:
        cmd += ["-n", "auto", "--maxprocesses=4"]
    run_command(cmd, env=env)


def benchmark_python(args: Any):
    env = get_python_env()

    # Define available device types per platform
    os_name = get_os()
    if os_name == "windows":
        device_types = ["d3d12", "vulkan", "cuda"]
    elif os_name == "linux":
        device_types = ["vulkan", "cuda"]
    elif os_name == "macos":
        device_types = ["metal"]
    else:
        raise RuntimeError(f"Unsupported OS for benchmarks: {os_name}")

    # If specific device is requested, only run for that device
    if args.device_type:
        if args.device_type == "nodevice":
            device_types = ["nodevice"]
        elif args.device_type not in device_types:
            print(f"Device type {args.device_type} not supported on {os_name}, skipping benchmarks")
            return
        else:
            device_types = [args.device_type]
    else:
        # Run for all device types plus nodevice tests
        device_types = device_types + ["nodevice"]

    try:
        # Lock GPU clocks
        if args.lock_gpu_clocks:
            cmd = ["python", str(PROJECT_DIR / "tools/gpu_clock.py"), "lock", "--ratio", "0.7"]
            if os_name == "linux":
                cmd = ["sudo"] + cmd
            run_command(cmd)

        # Run benchmarks for each device type
        for device_type in device_types:
            print(f"Running benchmarks for device type: {device_type}")

            cmd = ["pytest", "slangpy/benchmarks", "-ra", "--device-types", device_type]
            if args.mongodb_connection_string:
                cmd += ["--benchmark-upload", args.run_id]
                cmd += ["--benchmark-mongodb-connection-string", args.mongodb_connection_string]
                if args.mongodb_database_name:
                    cmd += ["--benchmark-mongodb-database-name", args.mongodb_database_name]

            try:
                run_command(cmd, env=env)
            except Exception as e:
                print(f"Benchmarks failed for device type {device_type}: {e}")
                if args.device_type:  # If specific device requested, fail hard
                    raise
                # Otherwise, continue with other devices
    finally:
        # Unlock GPU clocks
        if args.lock_gpu_clocks:
            cmd = ["python", str(PROJECT_DIR / "tools/gpu_clock.py"), "unlock"]
            if os_name == "linux":
                cmd = ["sudo"] + cmd
            run_command(cmd)


def coverage_report(args: Any):
    if not "coverage" in args.flags:
        print("Coverage flag not set, skipping coverage report.")
    os.makedirs("reports", exist_ok=True)
    run_command(["gcovr", "-r", ".", "-f", "src/sgl", "--html", "reports/coverage.html"])


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

    parser_benchmark_python = commands.add_parser(
        "benchmark-python", help="run benchmarks (python)"
    )
    parser_benchmark_python.add_argument("-r", "--run-id", type=str, required=True, help="Run ID")
    parser_benchmark_python.add_argument(
        "-c", "--mongodb-connection-string", type=str, help="MongoDB connection string"
    )
    parser_benchmark_python.add_argument(
        "-d", "--mongodb-database-name", type=str, help="MongoDB database name"
    )
    parser_benchmark_python.add_argument(
        "--device-type",
        type=str,
        help="Specific device type to benchmark (d3d12, vulkan, cuda, metal, nodevice)",
    )
    parser_benchmark_python.add_argument(
        "--lock-gpu-clocks",
        action="store_true",
        help="Lock GPU clocks during benchmarking (requires admin privileges).",
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
        "benchmark-python": benchmark_python,
        "coverage-report": coverage_report,
    }[args.command](args)

    return 0


if __name__ == "__main__":
    main()
