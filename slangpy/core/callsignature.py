# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any, Optional

from slangpy.core.native import CallMode

import slangpy.bindings.typeregistry as tr
from slangpy import ModifierID, TypeReflection
from slangpy.bindings.marshall import BindContext, ReturnContext
from slangpy.bindings.boundvariable import BoundCall, BoundVariable
from slangpy.builtin.value import NoneMarshall
from slangpy.reflection.reflectiontypes import (
    SlangFunction,
    SlangType,
)
from slangpy.reflection.typeresolution import resolve_function, ResolvedParam, ResolutionDiagnostic
from slangpy.types import Tensor
from slangpy.types.valueref import ValueRef

if TYPE_CHECKING:
    from slangpy.core.function import FunctionBuildInfo


class MismatchReason:
    def __init__(self, reason: str):
        super().__init__()
        self.reason = reason


class ResolveException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


from slangpy.core.generator import (
    KernelGenException,
    generate_constants,
    generate_code,
)  # noqa: F401


# This detects if a type is a vector with its length defined by a generic
# parameter. As exceptions can be raised attempting to read col count
# of generic vectors, we handle the exception as a generic vector.


def is_generic_vector(type: TypeReflection) -> bool:
    if type.kind != TypeReflection.Kind.vector:
        return False
    try:
        if type.scalar_type != TypeReflection.Kind.none and type.col_count > 0:  # @IgnoreException
            return False
    except Exception:
        return True
    return True


def specialize(
    context: BindContext,
    signature: BoundCall,
    function: SlangFunction,
    diagnostics: ResolutionDiagnostic,
    this_type: Optional[SlangType],
):
    return resolve_function(context, function, signature, diagnostics, this_type)


def bind(
    context: BindContext, signature: BoundCall, function: SlangFunction, params: list[ResolvedParam]
) -> BoundCall:
    """
    Apply a matched signature to a slang function, adding slang type marshalls
    to the signature nodes and performing other work that kicks in once
    match has occured.
    """

    res = signature
    res.bind(function)

    for x in signature.args:
        b = x
        if x.param_index == len(params):
            assert function.return_type is not None
            b.bind(function.return_type, {ModifierID.out}, "_result")
        elif x.param_index == -1:
            assert function.this is not None
            b.bind(
                function.this,
                {ModifierID.inout if function.mutating else ModifierID.inn},
                "_this",
            )
        else:
            b.bind(params[x.param_index])

    for k, v in signature.kwargs.items():
        b = v
        if k == "_result":
            assert function.return_type is not None
            b.bind(function.return_type, {ModifierID.out}, "_result")
        elif k == "_this":
            assert function.this is not None
            b.bind(
                function.this,
                {ModifierID.inout if function.mutating else ModifierID.inn},
                "_this",
            )
        else:
            b.bind(params[v.param_index])

    return res


def apply_explicit_vectorization(
    context: BindContext, call: BoundCall, args: tuple[Any, ...], kwargs: dict[str, Any]
):
    """
    Apply user supplied explicit vectorization options to the python variables.
    """
    call.apply_explicit_vectorization(context, args, kwargs)
    return call


def apply_implicit_vectorization(context: BindContext, call: BoundCall):
    """
    Apply implicit vectorization rules and calculate per variable dimensionality
    """
    call.apply_implicit_vectorization(context)
    return call


def finalize_mappings(context: BindContext, call: BoundCall):
    """
    Once overall call dimensionality is known, calculate any explicit
    mappings for variables that only have explicit types
    """
    call.finalize_mappings(context)
    return call


def calculate_differentiability(context: BindContext, call: BoundCall):
    """
    Calculate differentiability of all variables
    """
    for arg in call.args:
        arg.calculate_differentiability(context)
    for arg in call.kwargs.values():
        arg.calculate_differentiability(context)


def calculate_direct_binding(call: BoundCall):
    """
    Calculate direct binding eligibility for all variables.
    """
    for arg in call.args:
        arg.calculate_direct_bind()
    for arg in call.kwargs.values():
        arg.calculate_direct_bind()


def estimate_entrypoint_arguments_size(call: BoundCall, call_dimensionality: int) -> int:
    """
    Estimate the required entry point uniform byte size if the bound call where
    to bind all depth-0 bound variables and plus metadata fields
    (_thread_count, shape arrays) directly as entry point arguments.

    Note: This is currently an estimate, as the actual calculation really needs to
    take into account descriptors etc.

    :param call: The bound call containing all args/kwargs.
    :param call_dimensionality: The call dimensionality (determines shape array count).
    :return: Total inline-uniform size in bytes.
    """
    total = 0

    for node in call.values():
        # PackedArg types use ParameterBlock — excluded from inline accounting
        if node.create_param_block:
            continue
        if node.vector_type is not None:
            total += node.vector_type.uniform_layout.size
        # If vector_type is None (shouldn't happen after binding), skip safely

    # _thread_count: uint3 = 12 bytes
    total += 12

    # Shape arrays: _grid_stride, _grid_dim, _call_dim — each is int[call_dimensionality]
    if call_dimensionality > 0:
        total += call_dimensionality * 4 * 3  # 3 arrays × N × sizeof(int)

    return total


def calculate_call_dimensionality(signature: BoundCall) -> int:
    """
    Calculate the dimensionality of the call
    """
    dimensionality = 0
    nodes: list[BoundVariable] = []
    for node in signature.values():
        node.get_input_list(nodes)
    for input in nodes:
        if input.call_dimensionality is not None:
            dimensionality = max(dimensionality, input.call_dimensionality)
    return dimensionality


def create_return_value_binding(context: BindContext, signature: BoundCall, return_type: Any):
    """
    Create the return value for the call
    """

    # If return values are not needed or already set, early out
    if context.call_mode != CallMode.prim:
        return
    node = signature.kwargs.get("_result")
    if node is None or not isinstance(node.python, NoneMarshall):
        return

    # Should have an explicit vector type by now.
    assert node.vector_type is not None

    # If no desired return type was specified explicitly, fill in a useful default
    if return_type is None:
        if context.call_dimensionality == 0:
            return_type = ValueRef
        else:
            return_type = Tensor

    return_ctx = ReturnContext(node.vector_type, context)
    python_type = tr.get_or_create_type(context.layout, return_type, return_ctx)

    node.call_dimensionality = context.call_dimensionality
    node.python = python_type
