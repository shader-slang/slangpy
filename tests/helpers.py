# SPDX-License-Identifier: Apache-2.0

import sgl
import sys
from pathlib import Path

SHADER_DIR = Path(__file__).parent

if sys.platform == "win32":
    DEFAULT_DEVICE_TYPES = [sgl.DeviceType.d3d12, sgl.DeviceType.vulkan]
elif sys.platform == "linux" or sys.platform == "linux2":
    DEFAULT_DEVICE_TYPES = [sgl.DeviceType.vulkan]
elif sys.platform == "darwin":
    DEFAULT_DEVICE_TYPES = [sgl.DeviceType.vulkan]
else:
    raise RuntimeError("Unsupported platform")

DEVICE_CACHE = {}

def get_device(type: sgl.DeviceType, use_cache: bool = True) -> sgl.Device:
    if use_cache and type in DEVICE_CACHE:
        return DEVICE_CACHE[type]
    device = sgl.Device(
        type=type,
        enable_debug_layers=True,
        compiler_options={
            "include_paths": [SHADER_DIR],
            "debug_info": sgl.SlangDebugInfoLevel.standard,
        },
    )
    if use_cache:
        DEVICE_CACHE[type] = device
    return device

