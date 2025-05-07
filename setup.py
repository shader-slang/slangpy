# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- coding: utf-8 -*-

from __future__ import print_function

import sys, re, os, subprocess, shutil
from pathlib import Path

try:
    from setuptools import Extension, setup, find_packages
    from setuptools.command.build_ext import build_ext
except ImportError:
    print(
        "The preferred way to invoke 'setup.py' is via pip, as in 'pip "
        "install .'. If you wish to run the setup script directly, you must "
        "first install the build dependencies listed in pyproject.toml!",
        file=sys.stderr,
    )
    raise

SOURCE_DIR = Path(__file__).parent.resolve()

if sys.platform.startswith("win"):
    PLATFORM = "windows"
elif sys.platform.startswith("linux"):
    PLATFORM = "linux"
elif sys.platform.startswith("darwin"):
    PLATFORM = "macos"
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

CMAKE_PRESET = {
    "windows": "windows-msvc",
    "linux": "linux-gcc",
    "macos": "macos-arm64-clang",
}[PLATFORM]


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Setup environment variables
        env = os.environ.copy()
        if os.name == "nt":
            sys.path.append(str(Path(__file__).parent / "tools"))
            import msvc  # type: ignore

            env = msvc.msvc14_get_vc_env("x64")

        build_dir = str(SOURCE_DIR / "build/pip")

        cmake_args = [
            "--preset",
            CMAKE_PRESET,
            "-B",
            build_dir,
            "-DCMAKE_DEFAULT_BUILD_TYPE=Release",
            f"-DPython_ROOT_DIR:PATH={sys.prefix}",
            f"-DPython_FIND_REGISTRY:STRING=NEVER",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            f"-DCMAKE_INSTALL_LIBDIR=.",
            f"-DCMAKE_INSTALL_BINDIR=.",
            f"-DCMAKE_INSTALL_INCLUDEDIR=include",
            f"-DCMAKE_INSTALL_DATAROOTDIR=.",
            "-DSGL_BUILD_EXAMPLES=OFF",
            "-DSGL_BUILD_TESTS=OFF",
        ]

        # Adding CMake arguments set as environment variable
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Configure, build and install
        subprocess.run(["cmake", *cmake_args], env=env, check=True)
        subprocess.run(["cmake", "--build", build_dir], env=env, check=True)
        subprocess.run(["cmake", "--install", build_dir], env=env, check=True)

        # Remove files that are not needed
        for file in ["slang-rhi.lib"]:
            path = extdir / file
            if path.exists():
                os.remove(path)


VERSION_REGEX = re.compile(r"^\s*#\s*define\s+SGL_VERSION_([A-Z]+)\s+(.*)$", re.MULTILINE)

with open("src/sgl/sgl.h") as f:
    matches = dict(VERSION_REGEX.findall(f.read()))
    version = "{MAJOR}.{MINOR}.{PATCH}".format(**matches)
    print(f"version={version}")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    version=version,
    packages=find_packages(),
    package_data={
        "slangpy": ["slang/*.slang"],
    },
    include_package_data=True,
    ext_modules=[CMakeExtension("slangpy.slangpy_ext")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
