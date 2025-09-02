#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import platform
import shutil
import subprocess
import argparse


def find_nvidia_smi() -> str:
    """
    Locate the nvidia-smi utility.
    """
    if platform.system() == "Windows":
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            nvidia_smi = (
                "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
                % os.environ["systemdrive"]
            )
    else:
        nvidia_smi = "nvidia-smi"
    return nvidia_smi


NVIDIA_SMI = find_nvidia_smi()


def run_command(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True).strip()


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


def enumerate_gpu_clocks(device_index: int):
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


def list_gpu_clocks(device_index: int):
    """
    List all supported memory/gpu clock speeds.
    """
    print(f"Selected GPU: {get_gpu_name(device_index)}")
    clocks = enumerate_gpu_clocks(device_index)
    mem_clocks = sorted(list(set([clock[0] for clock in clocks])), reverse=True)
    gpu_clocks = sorted(list(set([clock[1] for clock in clocks])), reverse=True)
    print("Supported mem clocks:", mem_clocks)
    print("Supported gpu clocks:", gpu_clocks)


def lock_gpu_clocks(device_index: int, ratio: float, conservative: bool):
    """
    Lock GPU memory and graphics clocks to a specific ratio.
    """
    print(f"Selected GPU: {get_gpu_name(device_index)}")
    clocks = enumerate_gpu_clocks(device_index)

    max_mem_clock = max(clocks, key=lambda x: x[0])[0] if clocks else 0
    max_gpu_clock = max(clocks, key=lambda x: x[1])[1] if clocks else 0

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

    print(f"Selected mem clock: {locked_mem_clock} MHz ({locked_mem_clock / max_mem_clock:.1%}):")
    print(f"Selected gpu clock: {locked_gpu_clock} MHz ({locked_gpu_clock / max_gpu_clock:.1%}):")

    print("Locking mem clock:")
    cmd = [NVIDIA_SMI, "-i", str(device_index), f"--lock-memory-clocks={locked_mem_clock}"]
    print(run_command(cmd))
    print("Locking gpu clock")
    cmd = [NVIDIA_SMI, "-i", str(device_index), f"--lock-gpu-clocks={locked_gpu_clock}"]
    print(run_command(cmd))


def unlock_gpu_clocks(device_index: int):
    """
    Unlock GPU memory and graphics clocks.
    """
    print(f"Selected GPU: {get_gpu_name(device_index)}")
    print("Unlocking mem clock:")
    print(run_command([NVIDIA_SMI, "-i", str(device_index), "--reset-memory-clocks"]))
    print("Unlocking gpu clock:")
    print(run_command([NVIDIA_SMI, "-i", str(device_index), "--reset-gpu-clocks"]))


def main():
    parser = argparse.ArgumentParser(description="GPU clock utility")

    commands = parser.add_subparsers(dest="command", required=True, help="sub-command help")

    parser_list = commands.add_parser("list", help="List supported GPU clocks")
    parser_list.add_argument("--device", type=int, default=0, help="GPU device index")

    parser_lock = commands.add_parser("lock", help="Lock GPU clocks")
    parser_lock.add_argument("--device", type=int, default=0, help="GPU device index")
    parser_lock.add_argument("--ratio", type=float, default=0.7, help="Clock ratio")
    parser_lock.add_argument(
        "--conservative", action="store_true", help="Use conservative clock selection"
    )

    parser_unlock = commands.add_parser("unlock", help="Unlock GPU clocks")
    parser_unlock.add_argument("--device", type=int, default=0, help="GPU device index")

    args = parser.parse_args()

    if args.command == "list":
        list_gpu_clocks(args.device)
    elif args.command == "lock":
        lock_gpu_clocks(args.device, args.ratio, args.conservative)
    elif args.command == "unlock":
        unlock_gpu_clocks(args.device)


if __name__ == "__main__":
    main()
