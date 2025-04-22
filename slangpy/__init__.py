# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# pyright: reportUnusedImport=false
# isort: skip_file
from .core.utils import create_device
import runpy
import pathlib

# Version number
__version__ = "0.23.0"

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

# Torch integration
from .torchintegration import TORCH_ENABLED
if TORCH_ENABLED:
    from .torchintegration import TorchModule

# Debug options for call data gen
from .core.calldata import set_dump_generated_shaders, set_dump_slang_intermediates

# Core slangpy interface
from .core.function import Function
from .core.struct import Struct
from .core.module import Module
from .core.instance import (
    InstanceList,
    InstanceBuffer
)

# Py torch integration
from .torchintegration import *

# Get shader include path for slangpy
SHADER_PATH = str(pathlib.Path(__file__).parent.absolute() / "slang")

# Helper to create devices
