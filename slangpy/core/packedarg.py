# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, cast
from slangpy import Module, ShaderCursor
from slangpy.core.native import (
    unpack_arg,
    get_value_signature,
    NativeMarshall,
    CallMode,
    NativePackedArg,
)
from slangpy.bindings import get_or_create_type, PYTHON_TYPES, Marshall, BindContext
from slangpy.reflection import SlangType, SlangProgramLayout


class PackedArg(NativePackedArg):
    def __init__(self, module: Module, arg_value: Any):
        python = get_or_create_type(module.layout, type(arg_value), arg_value)
        value = python.build_shader_object(
            BindContext(module.layout, CallMode.prim, module.device_module, {}), arg_value
        )
        if value is None:
            raise ValueError(
                f"Cannot build shader object for {arg_value} of type {type(arg_value)}"
            )
        super().__init__(python, value)
        self.slangpy_signature = f"PACKED[{get_value_signature(arg_value)}]"
