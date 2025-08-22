# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pytest

from slangpy.core.calldata import SLANG_PATH

import slangpy
from slangpy import (
    Device,
    DeviceType,
    SlangCompilerOptions,
    SlangDebugInfoLevel,
    NativeHandle,
)

SHADER_DIR = Path(__file__).parent

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
                "include_paths": [SHADER_DIR, SLANG_PATH],
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
