# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from hashlib import sha256
from os import PathLike
from typing import Any, Mapping
import slangpy as spy
import sys
import pytest
from pathlib import Path

SHADER_DIR = Path(__file__).parent

if sys.platform == "win32":
    DEFAULT_DEVICE_TYPES = [
        spy.DeviceType.d3d12,
        spy.DeviceType.vulkan,
        spy.DeviceType.cuda,
    ]
elif sys.platform == "linux" or sys.platform == "linux2":
    DEFAULT_DEVICE_TYPES = [spy.DeviceType.vulkan, spy.DeviceType.cuda]
elif sys.platform == "darwin":
    DEFAULT_DEVICE_TYPES = [spy.DeviceType.metal]
else:
    raise RuntimeError("Unsupported platform")

ALL_SHADER_MODELS = [
    spy.ShaderModel.sm_6_0,
    spy.ShaderModel.sm_6_1,
    spy.ShaderModel.sm_6_2,
    spy.ShaderModel.sm_6_3,
    spy.ShaderModel.sm_6_4,
    spy.ShaderModel.sm_6_5,
    spy.ShaderModel.sm_6_6,
    spy.ShaderModel.sm_6_7,
]


def all_shader_models_from(shader_model: spy.ShaderModel) -> list[spy.ShaderModel]:
    return ALL_SHADER_MODELS[ALL_SHADER_MODELS.index(shader_model) :]


DEVICE_CACHE = {}


# Returns a unique random 16 character string for every variant of every test.
@pytest.fixture
def test_id(request: Any):
    return sha256(request.node.nodeid.encode()).hexdigest()[:16]


def get_device(type: spy.DeviceType, use_cache: bool = True) -> spy.Device:
    if use_cache and type in DEVICE_CACHE:
        return DEVICE_CACHE[type]
    device = spy.Device(
        type=type,
        enable_debug_layers=True,
        compiler_options={
            "include_paths": [SHADER_DIR],
            "debug_info": spy.SlangDebugInfoLevel.standard,
        },
    )
    if use_cache:
        DEVICE_CACHE[type] = device
    return device


def create_session(device: spy.Device, defines: Mapping[str, str]) -> spy.SlangSession:
    return device.create_slang_session(
        compiler_options={
            "include_paths": [SHADER_DIR],
            "defines": defines,
            "debug_info": spy.SlangDebugInfoLevel.standard,
        }
    )


class Context:
    buffers: dict[str, spy.Buffer]
    textures: dict[str, spy.Texture]

    def __init__(self):
        super().__init__()
        self.buffers = {}
        self.textures = {}


def dispatch_compute(
    device: spy.Device,
    path: PathLike[str],
    entry_point: str,
    thread_count: list[int],
    buffers: dict[str, Any] = {},
    textures: dict[str, spy.Texture] = {},
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
                args["struct_type"] = (
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
