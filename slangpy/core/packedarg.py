# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, cast
from slangpy import Module
from slangpy.core.native import (
    get_value_signature,
    CallMode,
    CallDataMode,
    NativePackedArg,
    unpack_arg,
)
from slangpy.bindings import get_or_create_type, BindContext
import hashlib


class PackedArg(NativePackedArg):
    """
    Represents an argument that has been efficiently packed into
    a shader object for use in later functionc alls.
    """

    def __init__(self, module: Module, python_object: Any):

        # Confusing terminology: use 'unpack_arg' to convert object to plain-old-type (dict/list etc)
        unpacked_obj = unpack_arg(python_object)

        # Get python marshall for the unpacked object
        python = get_or_create_type(module.layout, type(unpacked_obj), unpacked_obj)

        # Create a shader object from the python marshall and init native structure
        shader_object = python.build_shader_object(
            BindContext(
                module.layout, CallMode.prim, module.device_module, {}, CallDataMode.global_data
            ),
            unpacked_obj,
        )
        if shader_object is None:
            raise ValueError(
                f"Cannot build shader object for {python_object} of type {type(python_object)}"
            )
        super().__init__(python, shader_object, python_object)

        # Read full signature then turn into shorter hash
        full_signature = get_value_signature(python_object)
        signature_hash = hashlib.sha256(full_signature.encode("utf-8")).hexdigest()[:16]
        self.slangpy_signature = f"PK[H:{signature_hash}]"


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
