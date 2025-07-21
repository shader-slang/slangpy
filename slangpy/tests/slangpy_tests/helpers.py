# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, cast

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
    NativeHandle,
)
from slangpy.types.buffer import NDBuffer
from slangpy.core.function import Function

SHADER_DIR = Path(__file__).parent

if os.environ.get("SLANGPY_DEVICE", None) is not None:
    DEFAULT_DEVICE_TYPES = [DeviceType[os.environ["SLANGPY_DEVICE"]]]
elif sys.platform == "win32":
    DEFAULT_DEVICE_TYPES = [DeviceType.d3d12, DeviceType.vulkan, DeviceType.cuda]
elif sys.platform == "linux" or sys.platform == "linux2":
    DEFAULT_DEVICE_TYPES = [DeviceType.vulkan, DeviceType.cuda]
elif sys.platform == "darwin":
    # TODO: we don't run any slangpy tests on metal due to slang bugs for now
    DEFAULT_DEVICE_TYPES = [DeviceType.metal]
else:
    raise RuntimeError("Unsupported platform")

DEVICE_CACHE: dict[tuple[DeviceType, bool], Device] = {}

METAL_PARAMETER_BLOCK_SUPPORT: Optional[bool] = None

# Enable this to make tests just run on d3d12 for faster testing
# DEFAULT_DEVICE_TYPES = [DeviceType.d3d12]

# Always dump stuff when testing
slangpy.set_dump_generated_shaders(True)
# slangpy.set_dump_slang_intermediates(True)

# Returns a unique random 16 character string for every variant of every test.


@pytest.fixture
def test_id(request: Any):
    return hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]


# Helper to get device of a given type


def get_device(
    type: DeviceType,
    use_cache: bool = True,
    cuda_interop: bool = False,
    existing_device_handles: Optional[Sequence[NativeHandle]] = None,
) -> Device:
    # Early out if we know we don't have support for parameter blocks
    global METAL_PARAMETER_BLOCK_SUPPORT
    if type == DeviceType.metal and METAL_PARAMETER_BLOCK_SUPPORT == False:
        pytest.skip(
            "Metal device does not support parameter blocks (requires argument buffer tier 2)"
        )

    if existing_device_handles is not None and use_cache:
        raise ValueError(
            "Cannot use existing_device_handles with caching enabled. "
            "Please set use_cache=False if you want to use existing_device_handles."
        )

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
        existing_device_handles=existing_device_handles,
    )

    # slangpy dependens on parameter block support which is not available on all Metal devices
    METAL_PARAMETER_BLOCK_SUPPORT = device.has_feature(slangpy.Feature.parameter_block)
    if METAL_PARAMETER_BLOCK_SUPPORT == False:
        pytest.skip(
            "Metal device does not support parameter blocks (requires argument buffer tier 2)"
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
    spy_module.logger = Logger(level=LogLevel.info)
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
    module.logger = Logger(level=LogLevel.info)

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
        element = cursor[i].read()
        if cursor.element_type_layout.kind == TypeReflection.Kind.matrix:
            element = element.to_numpy()
        data = np.append(data, element)

    return data


def write_ndbuffer_from_numpy(buffer: NDBuffer, data: np.ndarray, element_count: int = 0):
    cursor = buffer.cursor()
    shape = np.prod(np.array(buffer.shape))

    if element_count == 0:
        if cursor.element_type_layout.kind == TypeReflection.Kind.scalar:
            element_count = 1
        elif cursor.element_type_layout.kind == TypeReflection.Kind.vector:
            element_count = cursor.element_type.col_count
        elif cursor.element_type_layout.kind == TypeReflection.Kind.matrix:
            element_count = cursor.element_type.row_count * cursor.element_type.col_count
        else:
            raise ValueError(
                f"element_count not set and type is not scalar or vector: {cursor.element_type_layout.kind}"
            )

    for i in range(shape):
        # Resolves warning that converting a single-element numpy array to a scalar is deprecated
        if cursor.element_type_layout.kind == TypeReflection.Kind.scalar:
            cursor[i].write(data[i])
        else:
            buffer_data = np.array(data[i * element_count : (i + 1) * element_count])
            if cursor.element_type_layout.kind == TypeReflection.Kind.matrix:
                buffer_data = buffer_data.reshape(
                    cursor.element_type.row_count, cursor.element_type.col_count
                )
            cursor[i].write(buffer_data)

    cursor.apply()
