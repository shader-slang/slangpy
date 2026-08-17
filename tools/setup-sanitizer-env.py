# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
from typing import Optional


def sanitizer_path(path: pathlib.Path) -> str:
    return str(path).replace("\\", "/")


def append_github_env(name: str, value: str) -> None:
    env_file = os.environ.get("GITHUB_ENV")
    if not env_file:
        print(f"{name}={value}")
        return
    with open(env_file, "a", encoding="utf-8") as file:
        file.write(f"{name}={value}\n")


def prepend_github_path(path: pathlib.Path) -> None:
    github_path = os.environ.get("GITHUB_PATH")
    if not github_path:
        print(f"PATH={path}{os.pathsep}{os.environ.get('PATH', '')}")
        return
    with open(github_path, "a", encoding="utf-8") as file:
        file.write(f"{path}\n")


def find_symbolizer() -> Optional[str]:
    clang = shutil.which("clang++")
    if clang:
        result = subprocess.run(
            [clang, "--print-prog-name=llvm-symbolizer"],
            check=True,
            capture_output=True,
            text=True,
        )
        symbolizer = shutil.which(result.stdout.strip())
        if symbolizer:
            return symbolizer

    return (
        shutil.which("llvm-symbolizer")
        or shutil.which("llvm-symbolizer-20")
        or shutil.which("llvm-symbolizer-19")
        or shutil.which("llvm-symbolizer-18")
        or shutil.which("llvm-symbolizer-17")
    )


def query_clang_runtime(runtime_name: str) -> Optional[pathlib.Path]:
    clang = shutil.which("clang++")
    if not clang:
        raise RuntimeError(f"Could not find clang++ while locating {runtime_name}.")

    result = subprocess.run(
        [clang, f"--print-file-name={runtime_name}"],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime_path = pathlib.Path(result.stdout.strip())
    return runtime_path.resolve() if runtime_path.is_file() else None


def find_clang_runtime(runtime_names: tuple[str, ...]) -> pathlib.Path:
    for runtime_name in runtime_names:
        runtime_path = query_clang_runtime(runtime_name)
        if runtime_path:
            return runtime_path
    raise RuntimeError(
        "Could not locate a Clang sanitizer runtime matching: " + ", ".join(runtime_names)
    )


def deploy_windows_asan_runtime(binary_dir: pathlib.Path) -> None:
    machine = os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64").lower()
    architecture = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    runtime = find_clang_runtime((f"clang_rt.asan_dynamic-{architecture}.dll",))
    binary_dir.mkdir(parents=True, exist_ok=True)
    destination = binary_dir / runtime.name
    shutil.copy2(runtime, destination)
    prepend_github_path(runtime.parent)
    print(f"Copied Windows ASan runtime to {destination}")


def configure_linux_preload(sanitizers: set[str]) -> None:
    asan_runtime_names = (
        "libclang_rt.asan.so",
        "libclang_rt.asan-x86_64.so",
        "libclang_rt.asan-aarch64.so",
    )
    ubsan_runtime_names = (
        "libclang_rt.ubsan.so",
        "libclang_rt.ubsan_standalone-x86_64.so",
        "libclang_rt.ubsan_standalone-aarch64.so",
    )
    tsan_runtime_names = (
        "libclang_rt.tsan.so",
        "libclang_rt.tsan-x86_64.so",
        "libclang_rt.tsan-aarch64.so",
    )

    runtime_paths: list[str] = []
    if "address" in sanitizers:
        # ASan must be the first runtime loaded into the unsanitized Python process.
        runtime_paths.append(str(find_clang_runtime(asan_runtime_names)))
    if "undefined" in sanitizers:
        # UBSan is often incorporated into ASan or linked as a module dependency.
        # Preload its standalone runtime when Clang exposes one.
        for runtime_name in ubsan_runtime_names:
            runtime_path = query_clang_runtime(runtime_name)
            if runtime_path:
                runtime_paths.append(str(runtime_path))
                break
    if "thread" in sanitizers:
        runtime_paths.append(str(find_clang_runtime(tsan_runtime_names)))
    existing_preload = os.environ.get("LD_PRELOAD")
    if existing_preload:
        runtime_paths.append(existing_preload)
    append_github_env("LD_PRELOAD", os.pathsep.join(runtime_paths))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Configure sanitizer environment variables for SlangPy tests."
    )
    parser.add_argument("--binary-dir", required=True, type=pathlib.Path)
    parser.add_argument("--os", required=True, choices=("windows", "linux", "macos"))
    parser.add_argument("--sanitizers", default="address,undefined")
    args = parser.parse_args()

    sanitizers = set(args.sanitizers.split(","))
    unsupported_sanitizers = sanitizers - {"address", "undefined", "thread"}
    if unsupported_sanitizers:
        parser.error("Unsupported sanitizers: " + ", ".join(sorted(unsupported_sanitizers)))
    if "address" in sanitizers and "thread" in sanitizers:
        parser.error("AddressSanitizer and ThreadSanitizer cannot be combined")
    if "thread" in sanitizers and args.os == "windows":
        parser.error("ThreadSanitizer is not supported on Windows")

    workspace = pathlib.Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd())).resolve()
    binary_dir = args.binary_dir.resolve()
    asan_suppressions = workspace / "tools" / "asan-suppressions.txt"

    symbolizer = find_symbolizer()
    if symbolizer:
        append_github_env("ASAN_SYMBOLIZER_PATH", symbolizer)

    asan_options = [
        "halt_on_error=1",
        "symbolize=1",
        "fast_unwind_on_malloc=0",
    ]

    # The Windows ASan runtime rejects interceptor_via_lib suppressions. The
    # suppression file contains only Unix library names, so do not pass it there.
    if args.os != "windows" and "address" in sanitizers:
        asan_options.append(f"suppressions={sanitizer_path(asan_suppressions)}")
    if args.os == "linux":
        configure_linux_preload(sanitizers)

    if args.os == "linux" and "address" in sanitizers:
        sanitizer_log_dir = binary_dir / "sanitizer-logs"
        sanitizer_log_dir.mkdir(parents=True, exist_ok=True)
        xdg_runtime_dir = binary_dir / "xdg-runtime"
        xdg_runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        xdg_runtime_dir.chmod(0o700)
        asan_options = [
            "detect_leaks=1",
            "protect_shadow_gap=0",
            "abort_on_error=1",
            *asan_options,
        ]
        append_github_env(
            "LSAN_OPTIONS",
            f"exitcode=0:log_path={sanitizer_path(sanitizer_log_dir / 'lsan.log')}",
        )
        append_github_env("SANITIZER_LOG_DIR", str(sanitizer_log_dir))
        append_github_env("XDG_RUNTIME_DIR", str(xdg_runtime_dir))

    if args.os == "windows" and "address" in sanitizers:
        deploy_windows_asan_runtime(binary_dir)
        # The sanitizer host lives outside the Python installation, so CPython
        # cannot reliably derive the standard-library paths from argv[0].
        if sys.prefix != sys.base_prefix:
            python_paths = [sysconfig.get_path("purelib")]
            if existing_pythonpath := os.environ.get("PYTHONPATH"):
                python_paths.append(existing_pythonpath)
            append_github_env("PYTHONPATH", os.pathsep.join(python_paths))
        append_github_env("PYTHONHOME", sys.base_prefix)

    if "address" in sanitizers:
        append_github_env("ASAN_OPTIONS", ":".join(asan_options))
    if "undefined" in sanitizers:
        append_github_env("UBSAN_OPTIONS", "print_stacktrace=1:halt_on_error=1")
    if "thread" in sanitizers:
        tsan_options = ["halt_on_error=1", "second_deadlock_stack=1"]
        if symbolizer:
            tsan_options.append(
                f"external_symbolizer_path={sanitizer_path(pathlib.Path(symbolizer))}"
            )
        append_github_env("TSAN_OPTIONS", ":".join(tsan_options))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
