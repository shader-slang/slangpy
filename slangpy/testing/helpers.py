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

SHADER_INCLUDE_PATHS = []

if os.environ.get("SLANGPY_DEVICE", None) is not None:
    DEFAULT_DEVICE_TYPES = [DeviceType[os.environ["SLANGPY_DEVICE"]]]
elif sys.platform == "win32":
    DEFAULT_DEVICE_TYPES = [DeviceType.d3d12, DeviceType.vulkan, DeviceType.cuda]
elif sys.platform == "linux" or sys.platform == "linux2":
    DEFAULT_DEVICE_TYPES = [DeviceType.vulkan, DeviceType.cuda]
elif sys.platform == "darwin":
    DEFAULT_DEVICE_TYPES = [DeviceType.metal]
else:
    raise RuntimeError("Unsupported platform")

DEVICE_CACHE: dict[tuple[DeviceType, bool], Device] = {}

METAL_PARAMETER_BLOCK_SUPPORT: Optional[bool] = None

# Always dump stuff when testing
slangpy.set_dump_generated_shaders(True)
# slangpy.set_dump_slang_intermediates(True)

# Returns a unique random 16 character string for every variant of every test.


@pytest.fixture
def test_id(request: Any):
    return hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]


def start_session(shader_include_paths: list[Path] = []):
    """Start a new test session. Typically called from pytest_sessionstart."""

    global SHADER_INCLUDE_PATHS
    SHADER_INCLUDE_PATHS = shader_include_paths

    # pytest's stdout/stderr capturing sometimes leads to bad file descriptor exceptions
    # when logging in sgl. By setting IGNORE_PRINT_EXCEPTION, we ignore those exceptions.
    slangpy.ConsoleLoggerOutput.IGNORE_PRINT_EXCEPTION = True


def finish_session():
    """Finish the current test session. Typically called from pytest_sessionfinish."""

    # After all tests finished, close remaining devices. This ensures they're
    # cleaned up before pytorch, avoiding crashes for devices that share context.

    # If torch enabled, sync all devices to ensure all operations are finished.
    try:
        import torch

        torch.cuda.synchronize()
    except ImportError:  # @IgnoreException
        pass

    # Close all devices that were created during the tests.
    for device in slangpy.Device.get_created_devices():
        print(f"Closing device on shutdown {device.desc.label}")
        device.close()


def setup_test():
    """Setup a new test. Typically called from pytest_runtest_setup."""
    pass


def teardown_test():
    """Teardown the current test. Typically called from pytest_runtest_teardown."""

    # Ensure any devices that aren't part of the device cache are cleaned up.
    for device in slangpy.Device.get_created_devices():
        if device.desc.label.startswith("cached-"):
            continue
        print(f"Closing leaked device {device.desc.label}")
        device.close()


# Helper to get device of a given type
def get_device(
    type: DeviceType,
    use_cache: bool = True,
    cuda_interop: bool = False,
    existing_device_handles: Optional[Sequence[NativeHandle]] = None,
    label: Optional[str] = None,
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

    selected_adaptor_luid = None

    # This lets you force tests to use a specific GPU locally.
    # if existing_device_handles is None:
    #     adaptors = Device.enumerate_adapters(type)
    #     selected_adaptor_luid = adaptors[0].luid
    #     for adapter in adaptors:
    #         if "5090" in adapter.name:
    #             selected_adaptor_luid = adapter.luid
    #             break

    if label is None:
        label = ""
        if use_cache:
            label = "cached-slangpy"
        else:
            label = "uncached-slangpy"
        label += f"-{type.name}"
        if cuda_interop:
            label += "-cuda-interop"

    cache_key = (type, cuda_interop)
    if use_cache and cache_key in DEVICE_CACHE:
        return DEVICE_CACHE[cache_key]
    device = Device(
        type=type,
        adapter_luid=selected_adaptor_luid,
        enable_debug_layers=True,
        compiler_options=SlangCompilerOptions(
            {
                "include_paths": [*SHADER_INCLUDE_PATHS, SLANG_PATH],
                "debug_info": SlangDebugInfoLevel.standard,
            }
        ),
        enable_cuda_interop=cuda_interop,
        existing_device_handles=existing_device_handles,
        label=label,
    )

    # slangpy dependens on parameter block support which is not available on all Metal devices
    if type == DeviceType.metal:
        METAL_PARAMETER_BLOCK_SUPPORT = device.has_feature(slangpy.Feature.parameter_block)
        if METAL_PARAMETER_BLOCK_SUPPORT == False:
            pytest.skip(
                "Metal device does not support parameter blocks (requires argument buffer tier 2)"
            )

    if use_cache:
        DEVICE_CACHE[cache_key] = device
    return device


TORCH_DEVICES: dict[str, Device] = {}


# Helper that gets a device that wraps the current torch cuda context.
# This is useful for testing the torch integration.
def get_torch_device(type: DeviceType) -> Device:
    import torch

    # For testing, comment this in to disable backwards passes running on other threads
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    torch.cuda.init()
    torch.cuda.current_device()
    torch.cuda.current_stream()

    id = f"cached-torch-{torch.cuda.current_device()}-{type}"
    if id in TORCH_DEVICES:
        return TORCH_DEVICES[id]

    handles = slangpy.get_cuda_current_context_native_handles()
    device = get_device(
        type,
        use_cache=False,
        existing_device_handles=handles,
        cuda_interop=True,
        label=id + f"-{handles[1]}",
    )
    TORCH_DEVICES[id] = device
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
            element = element.to_numpy()  # type: ignore
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
