# SPDX-License-Identifier: Apache-2.0

import hashlib
from typing import Any, Optional

import pytest
import kernelfunctions
import sys
from pathlib import Path

from kernelfunctions.backend import (
    Device, DeviceType, SlangCompilerOptions, SlangDebugInfoLevel,
    DeclReflection, TypeReflection)

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
DEFAULT_DEVICE_TYPES = [DeviceType.d3d12]


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
                "include_paths": [SHADER_DIR],
                "debug_info": SlangDebugInfoLevel.standard,
            }
        ),
    )
    device.run_garbage_collection()
    if use_cache:
        DEVICE_CACHE[type] = device
    return device


# Helper that creates a module from source (if not already loaded) and find / returns
# a kernel function for it. This helper supports nested functions and structs, e.g.
# create_function_from_module(device, "MyStruct.add_numbers", <src>).
def create_function_from_module(
    device: Device, func_name: str, module_source: str
) -> kernelfunctions.Function:
    module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )

    names = func_name.split(".")

    node = module.module_decl
    while len(names) > 1:
        name = names.pop(0)
        node = node.find_first_child_of_kind(DeclReflection.Kind.struct, name)
        if node is None:
            raise ValueError(f"Struct '{name}' not found in module {module.name}")

    name = names.pop(0)
    function = kernelfunctions.Function(module, name, node)

    return function


class FakeSlangType:

    def __init__(
        self,
        kind: TypeReflection.Kind,
        name: str,
        element_count: Optional[int] = None,
        element_type: Optional[TypeReflection] = None,
        row_count: Optional[int] = None,
        col_count: Optional[int] = None,
        scalar_type: Optional[TypeReflection.ScalarType] = None,
    ):
        super().__init__()
        self.kind = kind
        self.name = name
        self.element_count = element_count
        self.element_type = element_type
        self.row_count = row_count
        self.col_count = col_count
        self.scalar_type = scalar_type
