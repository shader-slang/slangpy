# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path
from typing import Callable, Optional
import re


class Version:
    def __init__(self, major: int, minor: int, patch: int):
        super().__init__()
        self.major = major
        self.minor = minor
        self.patch = patch


FileCallback = Callable[[str, Version], str]


class File:
    def __init__(self, path: Path, handler: FileCallback):
        super().__init__()
        self.path = path
        self.handler = handler
        self.initial_content = path.read_text()
        self.modified_content: Optional[str] = None

    def apply_version(self, version: Version):
        modified = self.handler(self.initial_content, version)
        if modified != self.initial_content:
            self.modified_content = modified
            return True
        else:
            return False

    def save(self):
        if self.modified_content:
            print(f"Modified {self.path}")
            self.path.write_text(self.modified_content)


def fix_pyproject(file: str, version: Version) -> str:
    # version = "0.19.1"
    file = re.sub(
        r"version\s*=\s*\"\d+\.\d+\.\d+\"",
        f'version = "{version.major}.{version.minor}.{version.patch}"',
        file,
    )
    return file


def fix_import_version(file: str, version: Version) -> str:
    # __version__ = "0.19.1"
    file = re.sub(
        r"__version__\s*=\s*\"\d+\.\d+\.\d+\"",
        f'__version__ = "{version.major}.{version.minor}.{version.patch}"',
        file,
    )
    return file


def fix_sgl_requirements(file: str, version: Version) -> str:
    # nv-sgl == 0.19.1
    file = re.sub(
        r"nv\-sgl\s*==\s*\d+\.\d+\.\d+",
        f'nv-sgl == {version.major}.{version.minor}.{version.patch}',
        file,
    )
    return file


def fix_slangpy_version(save: bool = False):

    version = Version(0, 1, 0)
    root = Path(__file__).parent.parent

    changlog = root / "docs/changelog.rst"
    changlog_content = changlog.read_text()

    # find last version in changlog
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", changlog_content)
    if match:
        version = Version(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    else:
        exit(1)

    # run files patch
    files = [
        File(root / "pyproject.toml", fix_pyproject),
        File(root / "slangpy/__init__.py", fix_import_version),
    ]
    for file in files:
        file.apply_version(version)

    if save:
        for file in files:
            file.save()


def fix_sgl_version(save: bool = False):

    version = Version(0, 1, 0)
    root = Path(__file__).parent.parent

    pyproj = root / "pyproject.toml"
    pyproj_content = pyproj.read_text()

    # find sgl version number in pyproject.toml
    match = re.search(r"nv\-sgl==(\d+)\.(\d+)\.(\d+)", pyproj_content)
    if match:
        version = Version(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    else:
        exit(1)

    # store sgl version number in other requirements files
    files = [
        File(root / "requirements.txt", fix_sgl_requirements),
        File(root / "requirements-dev.txt", fix_sgl_requirements),

    ]
    for file in files:
        file.apply_version(version)

    if save:
        for file in files:
            file.save()


def run(save: bool = False):
    fix_slangpy_version(save)
    fix_sgl_version(save)


run(True)
