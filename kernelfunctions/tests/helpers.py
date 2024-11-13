# SPDX-License-Identifier: Apache-2.0

import hashlib
from types import NoneType
from typing import Any, Optional

import pytest
import kernelfunctions
import sys
from pathlib import Path

from kernelfunctions.backend import (
    Device, DeviceType, SlangCompilerOptions, SlangDebugInfoLevel,
    TypeReflection)
from kernelfunctions.calldata import SLANG_PATH
from kernelfunctions.module import Module
from kernelfunctions.typeregistry import PYTHON_TYPES, get_or_create_type
from kernelfunctions.core import BaseTypeImpl, Shape

SHADER_DIR = Path(__file__).parent

if sys.platform == "win32":
    DEFAULT_DEVICE_TYPES = [DeviceType.d3d12, DeviceType.vulkan]
elif sys.platform == "linux" or sys.platform == "linux2":
    DEFAULT_DEVICE_TYPES = [DeviceType.vulkan]
elif sys.platform == "darwin":
    DEFAULT_DEVICE_TYPES = [DeviceType.vulkan]
else:
    raise RuntimeError("Unsupported platform")

DEVICE_CACHE: dict[DeviceType, Device] = {}

# Enable this to make tests just run on d3d12 for faster testing
# DEFAULT_DEVICE_TYPES = [DeviceType.d3d12]


# Returns a unique random 16 character string for every variant of every test.
@pytest.fixture
def test_id(request: Any):
    return hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]


# Helper to get device of a given type
def get_device(type: DeviceType, use_cache: bool = True) -> Device:
    if use_cache and type in DEVICE_CACHE:
        return DEVICE_CACHE[type]
    device = Device(
        type=type,
        enable_debug_layers=True,
        compiler_options=SlangCompilerOptions(
            {
                "include_paths": [SHADER_DIR, SLANG_PATH],
                "debug_info": SlangDebugInfoLevel.standard,
            }
        ),
    )
    device.run_garbage_collection()
    if use_cache:
        DEVICE_CACHE[type] = device
    return device

# Helper that creates a module from source (if not already loaded) and returns
# the corresponding slangpy module.


def create_module(
    device: Device, module_source: str
) -> kernelfunctions.Module:
    module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )
    return kernelfunctions.Module(module)

# Helper that creates a module from source (if not already loaded) and find / returns
# a kernel function for it. This helper supports nested functions and structs, e.g.
# create_function_from_module(device, "MyStruct.add_numbers", <src>).


def create_function_from_module(
    device: Device, func_name: str, module_source: str, options: dict[str, Any] = {}
) -> kernelfunctions.Function:

    if not 'import "slangpy";' in module_source:
        module_source = 'import "slangpy";\n' + module_source

    slang_module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )
    module = Module(slang_module, options=options)

    names = func_name.split(".")

    if len(names) == 1:
        function = module.find_function(names[0])
    else:
        type_name = "::".join(names[:-1])
        function = module.find_function_in_struct(type_name, names[-1])
    if function is None:
        raise ValueError(f"Could not find function {func_name}")
    return function
