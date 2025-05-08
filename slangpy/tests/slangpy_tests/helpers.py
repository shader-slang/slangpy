# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import numpy as np

from slangpy.core.calldata import SLANG_PATH

import slangpy
from slangpy import Module
from slangpy import (
    Device,
    DeviceType,
    SlangCompilerOptions,
    SlangDebugInfoLevel,
    TypeReflection,
    Logger,
    LogLevel,
)
from slangpy.types.buffer import NDBuffer
from slangpy.core.function import Function

SHADER_DIR = Path(__file__).parent

if os.environ.get("SLANGPY_DEVICE", None) is not None:
    DEFAULT_DEVICE_TYPES = [DeviceType[os.environ["SLANGPY_DEVICE"]]]
elif sys.platform == "win32":
    DEFAULT_DEVICE_TYPES = [DeviceType.d3d12, DeviceType.vulkan]
elif sys.platform == "linux" or sys.platform == "linux2":
    DEFAULT_DEVICE_TYPES = [DeviceType.vulkan]
elif sys.platform == "darwin":
    # TODO: we don't run any slangpy tests on metal due to slang bugs for now
    DEFAULT_DEVICE_TYPES = []  # [DeviceType.metal]
else:
    raise RuntimeError("Unsupported platform")

DEVICE_CACHE: dict[tuple[DeviceType, bool], Device] = {}

# Enable this to make tests just run on d3d12 for faster testing
# DEFAULT_DEVICE_TYPES = [DeviceType.d3d12]

# Always dump stuff when testing
slangpy.set_dump_generated_shaders(True)

# Returns a unique random 16 character string for every variant of every test.


@pytest.fixture
def test_id(request: Any):
    return hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]


# Helper to get device of a given type


def get_device(type: DeviceType, use_cache: bool = True, cuda_interop: bool = False) -> Device:
    cache_key = (type, cuda_interop)
    if use_cache and cache_key in DEVICE_CACHE:
        return DEVICE_CACHE[cache_key]
    device = Device(
        type=type,
        enable_debug_layers=True,
        compiler_options=SlangCompilerOptions(
            {
                "include_paths": [SHADER_DIR, SLANG_PATH],
                "debug_info": SlangDebugInfoLevel.standard,
            }
        ),
        enable_cuda_interop=cuda_interop,
    )
    if use_cache:
        DEVICE_CACHE[cache_key] = device
    return device


# Helper that creates a module from source (if not already loaded) and returns
# the corresponding slangpy module.


def create_module(
    device: Device,
    module_source: str,
    link: list[Any] = [],
    options: dict[str, Any] = {},
) -> slangpy.Module:
    module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )
    spy_module = slangpy.Module(module, link=link, options=options)
    spy_module.logger = Logger(level=LogLevel.debug)
    return spy_module


# Helper that creates a module from source (if not already loaded) and find / returns
# a kernel function for it. This helper supports nested functions and structs, e.g.
# create_function_from_module(device, "MyStruct.add_numbers", <src>).


def create_function_from_module(
    device: Device,
    func_name: str,
    module_source: str,
    link: list[Any] = [],
    options: dict[str, Any] = {},
) -> slangpy.Function:

    if not 'import "slangpy";' in module_source:
        module_source = 'import "slangpy";\n' + module_source

    slang_module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )
    module = Module(slang_module, link=link, options=options)
    module.logger = Logger(level=LogLevel.debug)

    names = func_name.split(".")

    if len(names) == 1:
        function = module.find_function(names[0])
    else:
        type_name = "::".join(names[:-1])
        function = module.find_function_in_struct(type_name, names[-1])
    if function is None:
        raise ValueError(f"Could not find function {func_name}")
    return cast(Function, function)


def read_ndbuffer_from_numpy(buffer: NDBuffer) -> np.ndarray:
    cursor = buffer.cursor()
    data = np.array([])
    shape = np.prod(np.array(buffer.shape))
    for i in range(shape):
        data = np.append(data, cursor[i].read())

    return data


def write_ndbuffer_from_numpy(buffer: NDBuffer, data: np.ndarray, element_count: int = 0):
    cursor = buffer.cursor()
    shape = np.prod(np.array(buffer.shape))

    if element_count == 0:
        if cursor.element_type_layout.kind == TypeReflection.Kind.scalar:
            element_count = 1
        elif cursor.element_type_layout.kind == TypeReflection.Kind.vector:
            element_count = cursor.element_type.col_count
        else:
            raise ValueError(
                f"element_count not set and type is not scalar or vector: {cursor.element_type_layout.kind}"
            )

    for i in range(shape):
        buffer_data = np.array(data[i * element_count : (i + 1) * element_count])
        cursor[i].write(buffer_data)

    cursor.apply()
