# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, cast
from slangpy import Module, ShaderCursor
from slangpy.core.native import unpack_arg, get_value_signature, NativeMarshall, CallMode
from slangpy.bindings import get_or_create_type, PYTHON_TYPES, Marshall, BindContext
from slangpy.reflection import SlangType, SlangProgramLayout


class PackedArg:
    def __init__(self, module: Module, arg_value: Any):
        super().__init__()

        self.python = get_or_create_type(module.layout, type(arg_value), arg_value)
        self.slangpy_signature = get_value_signature(arg_value)
        self.value = self.python.build_shader_object(
            BindContext(module.layout, CallMode.prim, module.device_module, {}), arg_value
        )
        if self.value is None:
            raise ValueError(
                f"Cannot build shader object for {arg_value} of type {type(arg_value)}"
            )

    def uniforms(self):
        return self.value
