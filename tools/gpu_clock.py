#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import platform
import shutil
import subprocess
import argparse
from typing import Optional


def find_nvidia_smi() -> str:
    """
    Locate the nvidia-smi utility.
    """
    if platform.system() == "Windows":
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            nvidia_smi = (
                "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
                % os.environ.get("systemdrive", "C:")
            )
    else:
        nvidia_smi = "nvidia-smi"
    return nvidia_smi


NVIDIA_SMI = find_nvidia_smi()


def run_command(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True).strip()


def nvidia_smi_mutation_command(arguments: list[str]) -> list[str]:
    """
    Build an nvidia-smi command for an operation that changes GPU state.

    Linux CI grants ci-runner passwordless sudo only for the exact clock
    lock/reset argument forms used below. The Python process and read-only
    nvidia-smi queries remain unprivileged.
    """
    command = [NVIDIA_SMI, *arguments]
    if platform.system() == "Linux":
        command = ["sudo", "-n", "--", *command]
    return command


def get_gpu_name(device_index: int):
    """
    Return the name of the GPU.
    """
    return run_command(
        [
            NVIDIA_SMI,
            "-i",
            str(device_index),
            "--query-gpu=name",
            "--format=csv,noheader,nounits",
        ]
    )


def enumerate_gpu_clocks(device_index: int) -> list[tuple[int, int]]:
    """
    Return a list of all memory/gpu clock combinations.
    """
    clocks: list[tuple[int, int]] = []
    output = run_command(
        [
            NVIDIA_SMI,
            "-i",
            str(device_index),
            "--query-supported-clocks=memory,graphics",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in output.splitlines():
        memory, graphics = map(int, line.split(","))
        clocks.append((memory, graphics))
    return clocks


def get_current_clocks(device_index: int) -> tuple[int, int, int]:
    """
    Return current graphics clock, memory clock, and temperature.
    """
    output = run_command(
        [
            NVIDIA_SMI,
            "-i",
            str(device_index),
            "--query-gpu=clocks.current.graphics,clocks.current.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    graphics, memory, temp = map(int, output.split(","))
    return graphics, memory, temp


def list_gpu_clocks(device_index: int) -> None:
    """
    List all supported memory/gpu clock speeds.
    """
    print(f"Selected GPU: {get_gpu_name(device_index)}")
    clocks = enumerate_gpu_clocks(device_index)
    mem_clocks = sorted(list(set([clock[0] for clock in clocks])), reverse=True)
    gpu_clocks = sorted(list(set([clock[1] for clock in clocks])), reverse=True)
    print(f"Supported mem clocks: {mem_clocks}")
    print(f"Supported gpu clocks: {gpu_clocks}")

    current_graphics, current_memory, temp = get_current_clocks(device_index)
    print(f"Current graphics clock: {current_graphics} MHz")
    print(f"Current memory clock: {current_memory} MHz")
    print(f"Current temperature: {temp} C")


def lock_gpu_clocks(
    device_index: int, ratio: float, conservative: bool, dry_run: bool = False
) -> Optional[tuple[int, int]]:
    """
    Lock GPU memory and graphics clocks to a specific ratio of maximum.

    Args:
        device_index: GPU device index (0, 1, etc.)
        ratio: Target ratio of max clock speed (0.0 to 1.0)
        conservative: If True, only select clocks at or below the ratio
        dry_run: If True, print what would be done but don't execute

    Returns:
        Tuple of (locked_mem_clock, locked_gpu_clock) if successful, None on error.
    """
    print(f"Selected GPU: {get_gpu_name(device_index)}")
    clocks = enumerate_gpu_clocks(device_index)

    if not clocks:
        print("ERROR: No supported clock frequencies found")
        return None

    max_mem_clock = max(clocks, key=lambda x: x[0])[0]
    max_gpu_clock = max(clocks, key=lambda x: x[1])[1]

    print(f"Max mem clock: {max_mem_clock} MHz")
    print(f"Max gpu clock: {max_gpu_clock} MHz")

    locked_mem_clock = 0
    locked_gpu_clock = 0

    best_ratio_error = (float("inf"), float("inf"))
    for mem_clock, gpu_clock in clocks:
        mem_ratio_error = ratio - mem_clock / max_mem_clock
        gpu_ratio_error = ratio - gpu_clock / max_gpu_clock
        if conservative and (mem_ratio_error < 0 or gpu_ratio_error < 0):
            continue
        mem_ratio_error = abs(mem_ratio_error)
        gpu_ratio_error = abs(gpu_ratio_error)

        if mem_ratio_error <= best_ratio_error[0] and gpu_ratio_error <= best_ratio_error[1]:
            best_ratio_error = (mem_ratio_error, gpu_ratio_error)
            locked_mem_clock = mem_clock
            locked_gpu_clock = gpu_clock

    if locked_mem_clock == 0 or locked_gpu_clock == 0:
        print("ERROR: Could not find suitable clock combination")
        return None

    print(f"Selected mem clock: {locked_mem_clock} MHz ({locked_mem_clock / max_mem_clock:.1%})")
    print(f"Selected gpu clock: {locked_gpu_clock} MHz ({locked_gpu_clock / max_gpu_clock:.1%})")

    if dry_run:
        print("(dry run - not executing)")
        return (locked_mem_clock, locked_gpu_clock)

    print("Locking mem clock:")
    cmd = nvidia_smi_mutation_command(
        ["-i", str(device_index), f"--lock-memory-clocks={locked_mem_clock}"]
    )
    print(run_command(cmd))

    print("Locking gpu clock:")
    cmd = nvidia_smi_mutation_command(
        ["-i", str(device_index), f"--lock-gpu-clocks={locked_gpu_clock}"]
    )
    print(run_command(cmd))

    return (locked_mem_clock, locked_gpu_clock)


def unlock_gpu_clocks(device_index: int, dry_run: bool = False) -> None:
    """
    Unlock GPU memory and graphics clocks.
    """
    print(f"Selected GPU: {get_gpu_name(device_index)}")

    if dry_run:
        print("(dry run - not executing)")
        return

    print("Unlocking mem clock:")
    print(
        run_command(nvidia_smi_mutation_command(["-i", str(device_index), "--reset-memory-clocks"]))
    )
    print("Unlocking gpu clock:")
    print(run_command(nvidia_smi_mutation_command(["-i", str(device_index), "--reset-gpu-clocks"])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU clock utility for benchmark reproducibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    commands = parser.add_subparsers(dest="command", required=True, help="sub-command help")

    parser_list = commands.add_parser("list", help="List supported GPU clocks")
    parser_list.add_argument("--device", type=int, default=0, help="GPU device index")

    parser_lock = commands.add_parser("lock", help="Lock GPU clocks to a stable frequency")
    parser_lock.add_argument("--device", type=int, default=0, help="GPU device index")
    parser_lock.add_argument(
        "--ratio",
        type=float,
        default=0.7,
        help="Target ratio of max clock speed (default: 0.7)",
    )
    parser_lock.add_argument(
        "--conservative",
        action="store_true",
        help="Only select clocks at or below the target ratio",
    )
    parser_lock.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )

    parser_unlock = commands.add_parser("unlock", help="Unlock GPU clocks")
    parser_unlock.add_argument("--device", type=int, default=0, help="GPU device index")
    parser_unlock.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_gpu_clocks(args.device)
    elif args.command == "lock":
        lock_gpu_clocks(args.device, args.ratio, args.conservative, args.dry_run)
    elif args.command == "unlock":
        unlock_gpu_clocks(args.device, args.dry_run)


if __name__ == "__main__":
    main()
