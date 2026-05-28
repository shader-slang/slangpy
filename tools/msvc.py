# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Taken from: https://github.com/pypa/setuptools/blob/v73.0.1/setuptools/msvc.py
Stripped out all code other than required for msvc14_get_vc_env


Improved support for Microsoft Visual C++ compilers.

Known supported compilers:
--------------------------
Microsoft Visual C++ 14.X:
    Microsoft Visual C++ Build Tools 2015 (x86, x64, arm)
    Microsoft Visual Studio Build Tools 2017 (x86, x64, arm, arm64)
    Microsoft Visual Studio Build Tools 2019 (x86, x64, arm, arm64)

This may also support compilers shipped with compatible Visual Studio versions.
"""

from __future__ import annotations

import contextlib
import itertools
import platform
import subprocess
from os.path import isdir, isfile, join
from subprocess import CalledProcessError
from typing import TYPE_CHECKING


import distutils.errors

# https://github.com/python/mypy/issues/8166
if not TYPE_CHECKING and platform.system() == "Windows":
    import winreg
    from os import environ
else:
    # Mock winreg and environ so the module can be imported on this platform.

    class winreg:
        HKEY_USERS = None
        HKEY_CURRENT_USER = None
        HKEY_LOCAL_MACHINE = None
        HKEY_CLASSES_ROOT = None

    environ: dict[str, str] = dict()


def _msvc14_find_vc2015():
    """Python 3.8 "distutils/_msvccompiler.py" backport"""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"Software\Microsoft\VisualStudio\SxS\VC7",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_32KEY,
        )
    except OSError:
        return None, None

    best_version = 0
    best_dir = None
    with key:
        for i in itertools.count():
            try:
                v, vc_dir, vt = winreg.EnumValue(key, i)
            except OSError:
                break
            if v and vt == winreg.REG_SZ and isdir(vc_dir):
                try:
                    version = int(float(v))
                except (ValueError, TypeError):
                    continue
                if version >= 14 and version > best_version:
                    best_version, best_dir = version, vc_dir
    return best_version, best_dir


def _msvc14_find_vc2017():
    """Python 3.8 "distutils/_msvccompiler.py" backport

    Returns "15, path" based on the result of invoking vswhere.exe
    If no install is found, returns "None, None"

    The version is returned to avoid unnecessarily changing the function
    result. It may be ignored when the path is not None.

    If vswhere.exe is not available, by definition, VS 2017 is not
    installed.
    """
    root = environ.get("ProgramFiles(x86)") or environ.get("ProgramFiles")
    if not root:
        return None, None

    suitable_components = (
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "Microsoft.VisualStudio.Workload.WDExpress",
    )

    for component in suitable_components:
        # Workaround for `-requiresAny` (only available on VS 2017 > 15.6)
        with contextlib.suppress(CalledProcessError, OSError, UnicodeDecodeError):
            path = (
                subprocess.check_output(
                    [
                        join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe"),
                        "-latest",
                        "-prerelease",
                        "-requires",
                        component,
                        "-property",
                        "installationPath",
                        "-products",
                        "*",
                    ]
                )
                .decode(encoding="mbcs", errors="strict")
                .strip()
            )

            path = join(path, "VC", "Auxiliary", "Build")
            if isdir(path):
                return 15, path

    return None, None  # no suitable component found


PLAT_SPEC_TO_RUNTIME = {
    "x86": "x86",
    "x86_amd64": "x64",
    "x86_arm": "arm",
    "x86_arm64": "arm64",
}


def _msvc14_find_vcvarsall(plat_spec):
    """Python 3.8 "distutils/_msvccompiler.py" backport"""
    _, best_dir = _msvc14_find_vc2017()
    vcruntime = None

    if plat_spec in PLAT_SPEC_TO_RUNTIME:
        vcruntime_plat = PLAT_SPEC_TO_RUNTIME[plat_spec]
    else:
        vcruntime_plat = "x64" if "amd64" in plat_spec else "x86"

    if best_dir:
        vcredist = join(
            best_dir,
            "..",
            "..",
            "redist",
            "MSVC",
            "**",
            vcruntime_plat,
            "Microsoft.VC14*.CRT",
            "vcruntime140.dll",
        )
        try:
            import glob

            vcruntime = glob.glob(vcredist, recursive=True)[-1]
        except (ImportError, OSError, LookupError):
            vcruntime = None

    if not best_dir:
        best_version, best_dir = _msvc14_find_vc2015()
        if best_version:
            vcruntime = join(
                best_dir,
                "redist",
                vcruntime_plat,
                "Microsoft.VC140.CRT",
                "vcruntime140.dll",
            )

    if not best_dir:
        return None, None

    vcvarsall = join(best_dir, "vcvarsall.bat")
    if not isfile(vcvarsall):
        return None, None

    if not vcruntime or not isfile(vcruntime):
        vcruntime = None

    return vcvarsall, vcruntime


def _msvc14_get_vc_env(plat_spec):
    """Python 3.8 "distutils/_msvccompiler.py" backport"""
    if "DISTUTILS_USE_SDK" in environ:
        return {key.lower(): value for key, value in environ.items()}

    vcvarsall, vcruntime = _msvc14_find_vcvarsall(plat_spec)
    if not vcvarsall:
        raise distutils.errors.DistutilsPlatformError("Unable to find vcvarsall.bat")

    try:
        out = subprocess.check_output(
            'cmd /u /c "{}" {} && set'.format(vcvarsall, plat_spec),
            stderr=subprocess.STDOUT,
        ).decode("utf-16le", errors="replace")
    except subprocess.CalledProcessError as exc:
        raise distutils.errors.DistutilsPlatformError("Error executing {}".format(exc.cmd)) from exc

    env = {
        key.lower(): value
        for key, _, value in (line.partition("=") for line in out.splitlines())
        if key and value
    }

    if vcruntime:
        env["py_vcruntime_redist"] = vcruntime
    return env


def msvc14_get_vc_env(plat_spec):
    """
    Patched "distutils._msvccompiler._get_vc_env" for support extra
    Microsoft Visual C++ 14.X compilers.

    Set environment without use of "vcvarsall.bat".

    Parameters
    ----------
    plat_spec: str
        Target architecture.

    Return
    ------
    dict
        environment
    """

    # Always use backport from CPython 3.8
    try:
        return _msvc14_get_vc_env(plat_spec)
    except distutils.errors.DistutilsPlatformError as exc:
        _augment_exception(exc, 14.0)
        raise


def _augment_exception(exc, version, arch=""):
    """
    Add details to the exception message to help guide the user
    as to what action will resolve it.
    """
    # Error if MSVC++ directory not found or environment not set
    message = exc.args[0]

    if "vcvarsall" in message.lower() or "visual c" in message.lower():
        # Special error message if MSVC++ not installed
        tmpl = "Microsoft Visual C++ {version:0.1f} or greater is required."
        message = tmpl.format(**locals())
        msdownload = "www.microsoft.com/download/details.aspx?id=%d"
        if version == 9.0:
            if arch.lower().find("ia64") > -1:
                # For VC++ 9.0, if IA64 support is needed, redirect user
                # to Windows SDK 7.0.
                # Note: No download link available from Microsoft.
                message += ' Get it with "Microsoft Windows SDK 7.0"'
            else:
                # For VC++ 9.0 redirect user to Vc++ for Python 2.7 :
                # This redirection link is maintained by Microsoft.
                # Contact vspython@microsoft.com if it needs updating.
                message += " Get it from http://aka.ms/vcpython27"
        elif version == 10.0:
            # For VC++ 10.0 Redirect user to Windows SDK 7.1
            message += ' Get it with "Microsoft Windows SDK 7.1": '
            message += msdownload % 8279
        elif version >= 14.0:
            # For VC++ 14.X Redirect user to latest Visual C++ Build Tools
            message += (
                ' Get it with "Microsoft C++ Build Tools": '
                r"https://visualstudio.microsoft.com"
                r"/visual-cpp-build-tools/"
            )

    exc.args = (message,)
