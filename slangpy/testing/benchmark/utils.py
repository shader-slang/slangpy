# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional
from pathlib import Path
import subprocess
import platform
from datetime import datetime


def get_project_info() -> dict[str, Any]:
    import slangpy

    return {
        "name": "slangpy",
        "version": slangpy.SGL_VERSION,
        "slang_build_tag": slangpy.SLANG_BUILD_TAG,
    }


def get_machine_info() -> dict[str, Any]:
    return {
        "node": platform.node(),
        "processor": platform.processor(),
        "machine": platform.machine(),
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
