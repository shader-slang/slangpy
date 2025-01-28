# SPDX-License-Identifier: Apache-2.0
# pyright: reportUnusedImport=false
# isort: skip_file
from .core.utils import create_device
import runpy
import pathlib

# Useful slangpy types
from . import types

# Bring all shared types into the top level namespace
from .types import *

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

# Core slangpy interface
from .core.function import Function
from .core.struct import Struct
from .core.module import Module
from .core.instance import (
    InstanceList,
    InstanceBuffer
)

# Get shader include path for slangpy
SHADER_PATH = str(pathlib.Path(__file__).parent.absolute() / "slang")

# Helper to create devices
