import os, sys

if os.name == "nt":
    package_dir = os.path.normpath(os.path.dirname(__file__))
    if os.path.exists(os.path.join(package_dir, "sgl.dll")):
        # This is a deployed package containing all the DLLs.
        # No need to setup a dll directory.
        pass
    elif os.path.exists(os.path.join(package_dir, ".build_dir")):
        # Loading package from a development build.
        # The DLLs are in the build directory.
        build_dir = open(os.path.join(package_dir, ".build_dir")).readline().strip()
        os.add_dll_directory(build_dir)
    else:
        print("Cannot locate sgl.dll.")
        sys.exit(1)

del os, sys

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
