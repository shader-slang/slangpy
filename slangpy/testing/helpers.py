# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Union, cast
from os import PathLike
import inspect

import pytest
import numpy as np

from slangpy.core.calldata import SLANG_PATH

import slangpy as spy
from slangpy import (
    Device,
    DeviceType,
    Buffer,
    Texture,
    Module,
    SlangCompilerOptions,
    SlangDebugInfoLevel,
    TypeReflection,
    Logger,
    LogLevel,
    NativeHandle,
)
from slangpy.types.buffer import NDBuffer
from slangpy.core.function import Function

# Global variables for device isolation. If SELECTED_DEVICE_TYPES is None, no restriction.
# If SELECTED_DEVICE_TYPES is an empty list, it means "nodevice" mode (only non-device tests).
# If SELECTED_DEVICE_TYPES has items, only tests for those device types will run.
SELECTED_DEVICE_TYPES: Optional[list[DeviceType]] = None

# Default device types based on the platform
if sys.platform == "win32":
    DEFAULT_DEVICE_TYPES = [DeviceType.d3d12, DeviceType.vulkan, DeviceType.cuda]
elif sys.platform == "linux" or sys.platform == "linux2":
    DEFAULT_DEVICE_TYPES = [DeviceType.vulkan, DeviceType.cuda]
elif sys.platform == "darwin":
    DEFAULT_DEVICE_TYPES = [DeviceType.metal]
else:
    raise RuntimeError("Unsupported platform")


# Called from pytest plugin if 'device-types' argument is provided
def set_device_types(device_types_str: Optional[str]) -> None:
    """Set the global device types. Called by pytest plugin."""
    global SELECTED_DEVICE_TYPES
    global DEFAULT_DEVICE_TYPES

    if device_types_str:
        if device_types_str == "nodevice":
            SELECTED_DEVICE_TYPES = []  # Empty list for nodevice mode
            DEFAULT_DEVICE_TYPES = []  # No device types for nodevice tests
        else:
            # Parse comma-separated device types
            device_type_names = [name.strip() for name in device_types_str.split(",")]
            SELECTED_DEVICE_TYPES = []
            for name in device_type_names:
                try:
                    SELECTED_DEVICE_TYPES.append(DeviceType[name])
                except KeyError:
                    raise ValueError(f"Invalid device type: {name}")
            DEFAULT_DEVICE_TYPES = SELECTED_DEVICE_TYPES.copy()
    else:
        SELECTED_DEVICE_TYPES = None  # No restriction


DEVICE_CACHE: dict[
    tuple[
        DeviceType,  # device_type
        tuple[Path, ...],  # include_paths
        tuple[NativeHandle, ...],  # existing_device_handles
        bool,  # cuda_interop
    ],
    Device,
] = {}

USED_TORCH_DEVICES: bool = False
METAL_PARAMETER_BLOCK_SUPPORT: Optional[bool] = None

# Always dump stuff when testing
spy.set_dump_generated_shaders(True)
# spy.set_dump_slang_intermediates(True)

# Returns a unique random 16 character string for every variant of every test.


@pytest.fixture
def test_id(request: Any):
    return hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]


def close_all_devices():
    # After all tests finished, close remaining devices. This ensures they're
    # cleaned up before pytorch, avoiding crashes for devices that share context.

    global USED_TORCH_DEVICES
    if USED_TORCH_DEVICES:
        import torch

        torch.cuda.synchronize()

    # Close all devices that were created during the tests.
    for device in Device.get_created_devices():
        print(f"Closing device on shutdown {device.desc.label}")
        device.close()


def close_leaked_devices():
    # Ensure any devices that aren't part of the device cache are cleaned up.
    for device in Device.get_created_devices():
        if device.desc.label.startswith("cached-"):
            continue
        print(f"Closing leaked device {device.desc.label}")
        device.close()


def should_skip_test_for_device(device_type: DeviceType) -> bool:
    """
    Check if a test should be skipped based on device filtering.
    Returns True if the test should be skipped.
    """
    if SELECTED_DEVICE_TYPES is None:
        return False  # No restriction, don't skip
    if len(SELECTED_DEVICE_TYPES) == 0:
        return True  # nodevice mode, skip all device tests
    return device_type not in SELECTED_DEVICE_TYPES


def should_skip_non_device_test() -> bool:
    """
    Check if a non-device test should be skipped based on device filtering.
    Non-device tests should only run when targeting 'nodevice' mode specifically.
    """
    if SELECTED_DEVICE_TYPES is None:
        return False  # No restriction, don't skip
    return len(SELECTED_DEVICE_TYPES) != 0  # Skip if specific devices were selected


# Helper to get device of a given type
def get_device(
    type: DeviceType,
    use_cache: bool = True,
    cuda_interop: bool = False,
    existing_device_handles: Optional[Sequence[NativeHandle]] = None,
    label: Optional[str] = None,
) -> Device:
    # Check if we're in device isolation mode and should restrict device types
    if SELECTED_DEVICE_TYPES is not None:
        if len(SELECTED_DEVICE_TYPES) == 0:
            raise RuntimeError(
                "get_device called when no device types are selected (nodevice mode)"
            )
        elif type not in SELECTED_DEVICE_TYPES:
            allowed_types = [dt.name for dt in SELECTED_DEVICE_TYPES]
            raise RuntimeError(
                f"get_device called with incompatible device type {type.name}, expected one of {allowed_types}"
            )

    # Early out if we know we don't have support for parameter blocks
    global METAL_PARAMETER_BLOCK_SUPPORT
    if type == DeviceType.metal and METAL_PARAMETER_BLOCK_SUPPORT == False:
        pytest.skip(
            "Metal device does not support parameter blocks (requires argument buffer tier 2)"
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

    # Use directory from caller module as the shader search path.
    caller_module_path: Optional[Path] = None
    stack_index = 1
    while caller_module_path == None:
        if Path(inspect.getfile(sys._getframe(stack_index))) != Path(__file__):
            caller_module_path = Path(inspect.getfile(sys._getframe(stack_index))).parent
        stack_index += 1
    include_paths = [caller_module_path, SLANG_PATH]

    cache_key = (
        type,
        tuple(include_paths),
        tuple(existing_device_handles) if existing_device_handles else (),
        cuda_interop,
    )

    if use_cache and cache_key in DEVICE_CACHE:
        return DEVICE_CACHE[cache_key]
    device = Device(
        type=type,
        adapter_luid=selected_adaptor_luid,
        enable_debug_layers=True,
        compiler_options=SlangCompilerOptions(
            {
                "include_paths": include_paths,
                "debug_info": SlangDebugInfoLevel.standard,
            }
        ),
        enable_cuda_interop=cuda_interop,
        existing_device_handles=existing_device_handles,
        label=label,
    )

    # slangpy dependens on parameter block support which is not available on all Metal devices
    if type == DeviceType.metal:
        METAL_PARAMETER_BLOCK_SUPPORT = device.has_feature(spy.Feature.parameter_block)
        if METAL_PARAMETER_BLOCK_SUPPORT == False:
            pytest.skip(
                "Metal device does not support parameter blocks (requires argument buffer tier 2)"
            )

    if use_cache:
        DEVICE_CACHE[cache_key] = device
    return device


# Helper that gets a device that wraps the current torch cuda context.
# This is useful for testing the torch integration.
def get_torch_device(type: DeviceType, use_cache: bool = True) -> Device:
    import torch

    global USED_TORCH_DEVICES
    USED_TORCH_DEVICES = True

    # For testing, comment this in to disable backwards passes running on other threads
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    torch.cuda.init()
    torch.cuda.current_device()
    torch.cuda.current_stream()

    handles = spy.get_cuda_current_context_native_handles()
    return get_device(
        type,
        use_cache=use_cache,
        existing_device_handles=handles,
        cuda_interop=True,
    )


# Helper that creates a module from source (if not already loaded) and returns
# the corresponding slangpy module.
def create_module(
    device: Device,
    module_source: str,
    link: list[Any] = [],
    options: dict[str, Any] = {},
) -> Module:
    module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )
    spy_module = Module(module, link=link, options=options)
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
) -> Function:

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


class Context:
    buffers: dict[str, Buffer]
    textures: dict[str, Texture]

    def __init__(self):
        super().__init__()
        self.buffers = {}
        self.textures = {}


def dispatch_compute(
    device: Device,
    path: Union[str, PathLike[str]],
    entry_point: str,
    thread_count: list[int],
    buffers: dict[str, Any] = {},
    textures: dict[str, Texture] = {},
    defines: dict[str, str] = {},
    compiler_options: "spy.SlangCompilerOptionsDict" = {},
    shader_model: spy.ShaderModel = spy.ShaderModel.sm_6_6,
) -> Context:
    # TODO(slang-rhi): Metal and CUDA don't support shader models.
    # we should move away from this concept and check features instead.
    if device.info.type == spy.DeviceType.metal or device.info.type == spy.DeviceType.cuda:
        shader_model = spy.ShaderModel.sm_6_0
    if shader_model > device.supported_shader_model:
        pytest.skip(f"Shader model {str(shader_model)} not supported")

    compiler_options["include_paths"] = device.slang_session.desc.compiler_options.include_paths
    compiler_options["shader_model"] = shader_model
    compiler_options["defines"] = defines
    compiler_options["debug_info"] = spy.SlangDebugInfoLevel.standard

    session = device.create_slang_session(compiler_options)
    program = session.load_program(module_name=str(path), entry_point_names=[entry_point])
    kernel = device.create_compute_kernel(program)

    ctx = Context()
    vars = {}
    params = {}

    for name, desc in buffers.items():
        is_global = kernel.reflection.find_field(name).is_valid()

        if isinstance(desc, spy.Buffer):
            buffer = desc
        else:
            args: Any = {
                "usage": spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            }
            if "size" in desc:
                args["size"] = desc["size"]
            if "element_count" in desc:
                args["element_count"] = desc["element_count"]
                args["resource_type_layout"] = (
                    kernel.reflection[name] if is_global else kernel.reflection[entry_point][name]
                )
            if "data" in desc:
                args["data"] = desc["data"]

            buffer = device.create_buffer(**args)

        ctx.buffers[name] = buffer

        if is_global:
            vars[name] = buffer
        else:
            params[name] = buffer

    for name, desc in textures.items():
        if isinstance(desc, spy.Texture):
            texture = desc
        else:
            raise NotImplementedError("Texture creation from dict not implemented")

        ctx.textures[name] = texture

        is_global = kernel.reflection.find_field(name).is_valid()
        if is_global:
            vars[name] = texture
        else:
            params[name] = texture

    kernel.dispatch(thread_count=thread_count, vars=vars, **params)

    return ctx
