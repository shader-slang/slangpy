import os, sys, platform, ctypes

package_dir = os.path.normpath(os.path.dirname(__file__))

if os.name == "nt":
    if os.path.exists(os.path.join(package_dir, "sgl.dll")):
        pass
    elif os.path.exists(os.path.join(package_dir, ".build_dir")):
        build_dir = open(os.path.join(package_dir, ".build_dir")).readline().strip()
        os.add_dll_directory(build_dir)
    else:
        print("Cannot locate sgl.dll.")
        sys.exit(1)
elif sys.platform == "darwin":
    # On macOS during development, pre-load dependent dylibs from the build dir
    # so the extension can resolve @rpath(libsgl.dylib) without requiring DYLD_LIBRARY_PATH.
    lib_here = os.path.join(package_dir, "libsgl.dylib")
    if not os.path.exists(lib_here) and os.path.exists(os.path.join(package_dir, ".build_dir")):
        try:
            build_dir = open(os.path.join(package_dir, ".build_dir")).readline().strip()
            # Preload likely dependencies if present
            for name in ("libslang.dylib", "libslang-rt.dylib", "libsgl.dylib"):
                p = os.path.join(build_dir, name)
                if os.path.exists(p):
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            # Fall through; the import may still work if libs are already discoverable
            pass

del os, sys, platform, ctypes

from importlib import import_module as _import

_import("slangpy.slangpy_ext")
del _import

# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# pyright: reportUnusedImport=false
# isort: skip_file
from .core.utils import create_device, create_torch_device
import runpy
import pathlib

# Useful slangpy types
from . import types

# Bring all shared types into the top level namespace
from .types import *

# Bring tested experimental types into top level namespace
from .experimental.gridarg import grid

# Slangpy reflection system
from . import reflection

# Required for extending slangpy
from . import bindings

# Trigger import of built in bindings so they get setup
from . import builtin as internal_marshalls

# Debug options for call data gen
from .core.calldata import (
    set_dump_generated_shaders,
    set_dump_slang_intermediates,
    set_print_generated_shaders,
)

# Core slangpy interface
from .core.function import Function
from .core.struct import Struct
from .core.module import Module
from .core.instance import InstanceList, InstanceBuffer
from .core.packedarg import pack

# Py torch integration
from .torchintegration import *

# Get shader include path for slangpy
SHADER_PATH = str(pathlib.Path(__file__).parent.absolute() / "slang")

# Helper to create devices
