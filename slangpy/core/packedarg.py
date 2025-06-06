# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, cast
from slangpy import Module
from slangpy.core.native import (
    get_value_signature,
    CallMode,
    NativePackedArg,
)
from slangpy.bindings import get_or_create_type, BindContext


class PackedArg(NativePackedArg):
    """
    Represents an argument that has been efficiently packed into
    a shader object for use in later functionc alls.
    """

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


def pack(module: Module, arg_value: Any) -> PackedArg:
    """
    Pack an argument for use in a shader call.

    :param module: Module used for type resolution.
    :param arg_value: The value to pack.
    :return: A PackedArg instance containing the packed argument.
    """
    if isinstance(arg_value, PackedArg):
        return cast(PackedArg, arg_value)
    return PackedArg(module, arg_value)
