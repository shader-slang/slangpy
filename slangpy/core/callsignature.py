# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Any, Optional

from slangpy.core.native import AccessType, CallMode, NativeMarshall

import slangpy.bindings.typeregistry as tr
import slangpy.reflection as slr
from slangpy.backend import FunctionReflection, ModifierID, TypeReflection
from slangpy.bindings.marshall import Marshall, BindContext, ReturnContext
from slangpy.bindings.boundvariable import (BoundCall, BoundVariable,
                                            BoundVariableException)
from slangpy.bindings.codegen import CodeGen
from slangpy.builtin.value import NoneMarshall, ValueMarshall
from slangpy.reflection.reflectiontypes import SlangFunction, SlangType
from slangpy.types.buffer import NDBuffer
from slangpy.types.valueref import ValueRef

if TYPE_CHECKING:
    from slangpy.core.function import FunctionBuildInfo


class MismatchReason:
    def __init__(self, reason: str):
        super().__init__()
        self.reason = reason


class KernelGenException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

# This detects if a type is a vector with its length defined by a generic
# parameter. As exceptions can be raised attempting to read col count
# of generic vectors, we handle the exception as a generic vector.


def is_generic_vector(type: TypeReflection) -> bool:
    if type.kind != TypeReflection.Kind.vector:
        return False
    try:
        if type.scalar_type != TypeReflection.Kind.none and type.col_count > 0:  # @IgnoreException
            return False
    except:
        return True
    return True


def specialize(
    context: BindContext,
    signature: BoundCall,
    functions: list[FunctionReflection],
    type: Optional[TypeReflection] = None
):
    # Handle current slang issue where init has to be found via ast, resulting in potential multiple functions
    if len(functions) > 1:
        if functions[0].name == "$init":
            matches = [x for x in functions if len(
                x.parameters) == signature.num_function_args]
            if len(matches) != 1:
                return MismatchReason("Overloaded $init functions are currently only supported if they have different argument counts.")
            function = matches[0]
        else:
            raise ValueError(
                "Internal error - Multiple function reflections provided, which should only happen with $init.")
    else:
        function = functions[0]

    # Expecting 'this' argument as first parameter of none-static member functions (except for $init)
    first_arg_is_this = type is not None and not function.has_modifier(
        ModifierID.static) and function.name != "$init"

    # Require '_result' argument for derivative calls, either as '_result' named parameter or last positional argument
    last_arg_is_retval = function.return_type is not None and function.return_type.name != 'void' and not "_result" in signature.kwargs and context.call_mode != CallMode.prim

    # Select the positional arguments we need to match against
    signature_args = signature.args
    if first_arg_is_this:
        signature_args[0].param_index = -1
        signature_args = signature_args[1:]
    if last_arg_is_retval:
        signature_args[-1].param_index = len(function.parameters)
        signature_args = signature_args[:-1]

    if signature.num_function_kwargs > 0 or signature.has_implicit_args:
        if function.is_overloaded:
            return MismatchReason("Call an overloaded function with named or implicit arguments is not currently supported.")

        function_parameters = [x for x in function.parameters]

        # Build empty positional list of python arguments to correspond to each slang argument
        positioned_args: list[Optional[BoundVariable]] = [
            None] * len(function_parameters)

        # Populate the first N arguments from provided positional arguments
        if len(signature_args) > len(function_parameters):
            return MismatchReason("Too many positional arguments.")
        for i, arg in enumerate(signature_args):
            positioned_args[i] = arg
            arg.param_index = i

        # Attempt to populate the remaining arguments from keyword arguments
        name_map = {param.name: i for i, param in enumerate(function_parameters)}
        for name, arg in signature.kwargs.items():
            if name == "_result":
                continue
            if name not in name_map:
                return MismatchReason(f"No parameter named '{name}'")
            i = name_map[name]
            if positioned_args[i] is not None:
                return MismatchReason(f"Parameter '{name}' is already specified as a positional argument.")
            positioned_args[i] = arg
            arg.param_index = i

        # Ensure all parameters are assigned
        if not all(x is not None for x in positioned_args):
            return MismatchReason("To use named or implicit arguments, all parameters must be specified.")

        # Choose either explicit vector type or slang type for specialization
        inputs: list[Any] = []
        for i, python_arg in enumerate(positioned_args):
            slang_param = function_parameters[i]
            assert python_arg is not None
            if python_arg.vector_type is not None:
                # Always take explicit vector types if provided
                inputs.append(python_arg.vector_type)
            elif is_generic_vector(slang_param.type):
                # HACK! Let types with a 'slang_element_type' try to resolve generic vector types
                # Failing that, fall back to python marshall
                sl_et = getattr(python_arg.python, 'slang_element_type', None)
                if isinstance(sl_et, slr.VectorType):
                    inputs.append(sl_et)
                else:
                    inputs.append(python_arg.python)
            elif slang_param.type.kind != TypeReflection.Kind.none:
                # If the type is fully resolved, use it
                inputs.append(slang_param.type)
            elif isinstance(python_arg.python, ValueMarshall) and python_arg.python.slang_type.name != 'Unknown':
                # If passing basic type to generic, resolve from its python type
                inputs.append(python_arg.python)
            else:
                return MismatchReason(
                    f"Parameter {i} is a generic or interface, so must either be passed a value type or have an explicit vector type.")
    else:
        # If no named or implicit arguments, just use explicit vector types for specialization
        inputs: list[Any] = [x.vector_type for x in signature_args]
        for i, arg in enumerate(signature_args):
            arg.param_index = i

    def to_type_reflection(input: Any) -> TypeReflection:
        if isinstance(input, NativeMarshall):
            return input.slang_type.type_reflection
        elif isinstance(input, TypeReflection):
            return input
        elif isinstance(input, str):
            return context.device_module.layout.find_type_by_name(input)
        elif isinstance(input, SlangType):
            return input.type_reflection
        else:
            raise KernelGenException(
                f"Cannot convert {input} to a TypeReflection for overload resolution.")

    input_types = [to_type_reflection(x) for x in inputs]
    if any(x is None for x in input_types):
        raise KernelGenException(
            "Unable to resolve all Slang types for specialization overload resolution.")

    specialized = function.specialize_with_arg_types(input_types)
    if specialized is None:
        return MismatchReason("No Slang overload found that matches the provided Python argument types.")

    return context.layout.find_function(specialized, type)


def validate_specialize(
    context: BindContext,
    signature: BoundCall,
    function: SlangFunction
):
    # Get sorted list of root parameters for trampoline function
    root_params = [y for y in sorted(signature.args + list(signature.kwargs.values()), key=lambda x: x.param_index)
                   if y.param_index >= 0 and y.param_index < len(function.parameters)]

    def to_type_reflection(input: Any) -> TypeReflection:
        if isinstance(input, Marshall):
            return input.slang_type.type_reflection
        elif isinstance(input, TypeReflection):
            return input
        elif isinstance(input, str):
            return context.device_module.layout.find_type_by_name(input)
        elif isinstance(input, SlangType):
            return input.type_reflection
        else:
            raise KernelGenException(
                f"After implicit casting, cannot convert {input} to TypeReflection.")

    types = [to_type_reflection(x.vector_type) for x in root_params]
    if any(x is None for x in types):
        raise KernelGenException(
            "After implicit casting, unable to resolve all Slang types for specialization overload resolution.")

    specialized = function.reflection.specialize_with_arg_types(types)
    if specialized is None:
        raise KernelGenException(
            "After implicit casting, no Slang overload found that matches the provided Python argument types.")


def bind(
        context: BindContext,
        signature: BoundCall,
        function: SlangFunction) -> BoundCall:
    """
    Apply a matched signature to a slang function, adding slang type marshalls
    to the signature nodes and performing other work that kicks in once
    match has occured.
    """

    res = signature
    res.bind(function)

    for x in signature.args:
        b = x
        if x.param_index == len(function.parameters):
            assert function.return_type is not None
            b.bind(function.return_type, {ModifierID.out}, '_result')
        elif x.param_index == -1:
            assert function.this is not None
            b.bind(function.this, {
                   ModifierID.inout if function.mutating else ModifierID.inn}, '_this')
        else:
            b.bind(function.parameters[x.param_index])

    for k, v in signature.kwargs.items():
        b = v
        if k == "_result":
            assert function.return_type is not None
            b.bind(function.return_type, {ModifierID.out}, '_result')
        elif k == "_this":
            assert function.this is not None
            b.bind(function.this, {
                   ModifierID.inout if function.mutating else ModifierID.inn}, '_this')
        else:
            b.bind(function.parameters[v.param_index])

    return res


def apply_explicit_vectorization(context: BindContext, call: BoundCall, args: tuple[Any, ...], kwargs: dict[str, Any]):
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
            return_type = NDBuffer

    return_ctx = ReturnContext(node.vector_type, context)
    python_type = tr.get_or_create_type(context.layout, return_type, return_ctx)

    node.call_dimensionality = context.call_dimensionality
    node.python = python_type


def generate_constants(build_info: 'FunctionBuildInfo', cg: CodeGen):
    if build_info.constants is not None:
        for k, v in build_info.constants.items():
            if isinstance(v, bool):
                cg.constants.append_statement(
                    f"export static const bool {k} = {'true' if v else 'false'}"
                )
            elif isinstance(v, (int, float)):
                cg.constants.append_statement(
                    f"export static const {type(v).__name__} {k} = {v}"
                )
            else:
                raise KernelGenException(
                    f"Constant value '{k}' must be an int, float or bool, not {type(v).__name__}"
                )


def generate_code(context: BindContext, build_info: 'FunctionBuildInfo', signature: BoundCall, cg: CodeGen):
    """
    Generate a list of call data nodes that will be used to generate the call
    """
    nodes: list[BoundVariable] = []

    # Generate the header
    cg.add_import("slangpy")
    cg.add_import(build_info.module.name)

    # Generate constants if specified
    generate_constants(build_info, cg)

    # Generate call data inputs if vector call
    call_data_len = context.call_dimensionality
    if call_data_len > 0:
        cg.call_data.append_statement(f"int[{call_data_len}] _call_stride")
        cg.call_data.append_statement(f"int[{call_data_len}] _call_dim")
    cg.call_data.append_statement(f"uint3 _thread_count")

    # Generate the context structure
    cg.context.type_alias("Context", f"ContextND<{context.call_dimensionality}>")

    # Generate call data definitions for all inputs to the kernel
    for node in signature.values():
        node.gen_call_data_code(cg, context)

    # Get sorted list of root parameters for trampoline function
    root_params = sorted(signature.values(), key=lambda x: x.param_index)

    # Generate the trampoline function
    trampoline_fn = "_trampoline"
    if context.call_mode != CallMode.prim:
        cg.trampoline.append_line("[Differentiable]")
    cg.trampoline.append_line(f"void {trampoline_fn}(Context context, CallData data)")
    cg.trampoline.begin_block()

    # Declare parameters and load inputs
    for x in root_params:
        assert x.vector_type is not None
        cg.trampoline.declare(x.vector_type.full_name, x.variable_name)
    for x in root_params:
        if x.access[0] == AccessType.read or x.access[0] == AccessType.readwrite:
            cg.trampoline.append_statement(
                f"data.{x.variable_name}.load(context.map(_m_{x.variable_name}), {x.variable_name})")

    cg.trampoline.append_indent()
    if any(x.variable_name == '_result' for x in root_params):
        cg.trampoline.append_code(f"_result = ")

    # Get function name, if it's the init function, use the result type
    func_name = build_info.name
    if func_name == "$init":
        results = [x for x in root_params if x.variable_name == '_result']
        assert len(results) == 1
        assert results[0].vector_type is not None
        func_name = results[0].vector_type.full_name
    elif len(root_params) > 0 and root_params[0].variable_name == '_this':
        func_name = f'_this.{func_name}'

    # Get the parameters that are not the result or this reference
    normal_params = [x for x in root_params if x.variable_name !=
                     '_result' and x.variable_name != '_this']

    # Internal call to the actual function
    cg.trampoline.append_code(
        f"{func_name}(" + ", ".join(x.variable_name for x in normal_params) + ");\n")

    # For each writable trampoline parameter, potentially store it
    for x in root_params:
        if x.access[0] == AccessType.write or x.access[0] == AccessType.readwrite or x.access[1] == AccessType.read:
            if not x.python.is_writable:
                raise BoundVariableException(
                    f"Cannot read back value for non-writable type", x)
            cg.trampoline.append_statement(
                f"data.{x.variable_name}.store(context.map(_m_{x.variable_name}), {x.variable_name})")

    cg.trampoline.end_block()
    cg.trampoline.append_line("")

    # Generate the main function
    cg.kernel.append_line('[shader("compute")]')
    cg.kernel.append_line("[numthreads(32, 1, 1)]")
    cg.kernel.append_line("void main(uint3 dispatchThreadID: SV_DispatchThreadID)")
    cg.kernel.begin_block()
    cg.kernel.append_statement(
        "if (any(dispatchThreadID >= call_data._thread_count)) return")

    # Loads / initializes call id
    context_args = "dispatchThreadID"
    if call_data_len > 0:
        cg.kernel.append_line(f"int[{call_data_len}] call_id = {{")
        cg.kernel.inc_indent()
        for i in range(call_data_len):
            cg.kernel.append_line(
                f"(dispatchThreadID.x/call_data._call_stride[{i}]) % call_data._call_dim[{i}],")
        cg.kernel.dec_indent()
        cg.kernel.append_statement("}")
        context_args += ", call_id"
    cg.kernel.append_statement(f"Context context = {{{context_args}}}")

    # Call the trampoline function
    fn = trampoline_fn
    if context.call_mode == CallMode.bwds:
        fn = f"bwd_diff({fn})"
    cg.kernel.append_statement(f"{fn}(context, call_data)")

    cg.kernel.end_block()
