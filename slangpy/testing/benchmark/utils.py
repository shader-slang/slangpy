# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional
from pathlib import Path
import os
import subprocess
import platform
from datetime import datetime
import shutil


def get_project_info() -> dict[str, Any]:
    import slangpy

    return {
        "name": "slangpy",
        "version": slangpy.SGL_VERSION,
        "slang_build_tag": slangpy.SLANG_BUILD_TAG,
    }


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


def run_command(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, universal_newlines=True
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {e.output}")
        raise e


def get_gpu_infos() -> list[dict[str, Any]]:
    infos = []

    try:
        output = run_command(
            [
                find_nvidia_smi(),
                "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,driver_version,name,gpu_serial,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
        )
        for line in output.split("\n"):
            values = line.strip().split(", ")

            infos.append(
                {
                    "index": int(values[0]),
                    "uuid": values[1],
                    "utilization": float(values[2]) / 100,
                    "memory_total": float(values[3]),
                    "memory_used": float(values[4]),
                    "driver_version": values[5],
                    "name": values[6],
                    "serial_number": values[7],
                    "clock_current_graphics": int(values[8]),
                    "clock_current_memory": int(values[9]),
                    "clock_max_graphics": int(values[10]),
                    "clock_max_memory": int(values[11]),
                    "temperature": float(values[12]),
                }
            )
    except Exception as e:
        print(f"Failed to retrieve GPU information: {e}")

    return infos


def get_machine_info() -> dict[str, Any]:
    return {
        "node": platform.node(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "gpus": get_gpu_infos(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "python_compiler": platform.python_compiler(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build": platform.python_build(),
    }


def in_any_parent(name: str, path: Optional[Path] = None):
    prev = None
    if not path:
        path = Path.cwd()
    while path and prev != path and not path.joinpath(name).exists():
        prev = path
        path = path.parent
    return path.joinpath(name).exists()


def subprocess_output(cmd: str) -> str:
    return subprocess.check_output(
        cmd.split(), stderr=subprocess.STDOUT, universal_newlines=True
    ).strip()


def get_commit_info() -> dict[str, Any]:
    dirty = False
    commit = "unversioned"
    commit_time = None
    author_time = None
    branch = "(unknown)"
    try:
        if in_any_parent(".git"):
            desc = subprocess_output("git describe --dirty --always --long --abbrev=40")
            desc = desc.split("-")
            if desc[-1].strip() == "dirty":
                dirty = True
                desc.pop()
            commit = desc[-1].strip("g")
            commit_time = subprocess_output('git show -s --pretty=format:"%cI"').strip('"')
            author_time = subprocess_output('git show -s --pretty=format:"%aI"').strip('"')
            branch = subprocess_output("git rev-parse --abbrev-ref HEAD")
            if branch == "HEAD":
                branch = "(detached head)"
        return {
            "id": commit,
            "time": datetime.fromisoformat(commit_time) if commit_time else None,
            "author_time": datetime.fromisoformat(author_time) if author_time else None,
            "dirty": dirty,
            "branch": branch,
        }
    except Exception as exc:
        return {
            "id": "unknown",
            "time": None,
            "author_time": None,
            "dirty": dirty,
            "error": (
                f"CalledProcessError({exc.returncode}, {exc.output!r})"
                if isinstance(exc, subprocess.CalledProcessError)
                else repr(exc)
            ),
            "branch": branch,
        }


def to_json(d: Any) -> Any:
    # Recurse through the dictionary and replace datetime objects with their ISO format
    if isinstance(d, dict):
        return {k: to_json(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_json(item) for item in d]
    elif isinstance(d, datetime):
        return d.isoformat()
    return d


def from_json(d: Any) -> Any:
    # Recurse through the dictionary and replace ISO format strings with datetime objects
    if isinstance(d, dict):
        return {k: from_json(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [from_json(item) for item in d]
    elif isinstance(d, str):
        try:
            return datetime.fromisoformat(d)
        except ValueError:
            return d
    return d


def _test_infos():
    import json

    print(json.dumps(to_json(get_machine_info()), indent=4))
    print(json.dumps(to_json(get_commit_info()), indent=4))
    print(json.dumps(to_json(get_project_info()), indent=4))


# _test_infos()
