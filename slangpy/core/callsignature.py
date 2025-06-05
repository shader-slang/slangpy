# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TYPE_CHECKING, Any, Optional

from slangpy.core.native import AccessType, CallMode, NativeMarshall

import slangpy.bindings.typeregistry as tr
import slangpy.reflection as slr
from slangpy import ModifierID, TypeReflection
from slangpy.bindings.marshall import Marshall, BindContext, ReturnContext
from slangpy.bindings.boundvariable import (
    BoundCall,
    BoundVariable,
    BoundVariableException,
)
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


class ResolveException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


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
    function: SlangFunction,
    this_type: Optional[SlangType] = None,
):
    # Special case for constructors
    if function.is_overloaded and function.is_constructor:
        matches = [
            x for x in function.overloads if len(x.parameters) == signature.num_function_args
        ]
        if len(matches) != 1:
            return MismatchReason(
                "Overloaded functions are currently only supported if they have different argument counts."
            )
        function = matches[0]

    # Expecting 'this' argument as first parameter of none-static member functions (except for constructors)
    first_arg_is_this = (
        this_type is not None and not function.static and not function.is_constructor
    )

    # Require '_result' argument for derivative calls, either as '_result' named parameter or last positional argument
    last_arg_is_retval = (
        function.return_type is not None
        and function.return_type.name != "void"
        and not "_result" in signature.kwargs
        and context.call_mode != CallMode.prim
    )

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
            return MismatchReason(
                f"Calling an overloaded function with named or implicit arguments is not currently supported."
            )

        function_parameters = [x for x in function.parameters]

        # Build empty positional list of python arguments to correspond to each slang argument
        positioned_args: list[Optional[BoundVariable]] = [None] * len(function_parameters)

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
                return MismatchReason(
                    f"Parameter '{name}' is already specified as a positional argument."
                )
            positioned_args[i] = arg
            arg.param_index = i

        # Ensure all parameters are assigned
        if not all(x is not None for x in positioned_args):
            return MismatchReason(
                "To use named or implicit arguments, all parameters must be specified."
            )

        # Choose either explicit vector type or slang type for specialization
        inputs: list[Any] = []
        for i, python_arg in enumerate(positioned_args):
            slang_param = function_parameters[i]
            assert python_arg is not None
            if python_arg.vector_type is not None:
                # Always take explicit vector types if provided
                inputs.append(python_arg.vector_type)
            elif (
                isinstance(slang_param.type, (slr.VectorType, slr.MatrixType, slr.ArrayType))
                and slang_param.type.is_generic
            ):
                # HACK! Let types with a 'slang_element_type' try to resolve known generic types
                # Failing that, fall back to python marshall
                sl_et = getattr(python_arg.python, "slang_element_type", None)
                if isinstance(sl_et, type(slang_param.type)):
                    inputs.append(sl_et)
                else:
                    inputs.append(python_arg.python)
            elif slang_param.type.type_reflection.kind != TypeReflection.Kind.none:
                # If the type is fully resolved, use it
                inputs.append(slang_param.type)
            elif (
                isinstance(python_arg.python, ValueMarshall)
                and python_arg.python.slang_type.name != "Unknown"
            ):
                # If passing basic type to generic, resolve from its python type
                inputs.append(python_arg.python)
            else:
                return MismatchReason(
                    f"Parameter {i} is a generic or interface, so must either be passed a value type or have an explicit vector type."
                )
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
                f"Cannot convert {input} to a TypeReflection for overload resolution."
            )

    input_types = [to_type_reflection(x) for x in inputs]
    if any(x is None for x in input_types):
        raise KernelGenException(
            "Unable to resolve all Slang types for specialization overload resolution."
        )

    specialized = function.reflection.specialize_with_arg_types(input_types)
    if specialized is None:
        return MismatchReason(
            "No Slang overload found that matches the provided Python argument types."
        )

    type_reflection = None if this_type is None else this_type.type_reflection

    return context.layout.find_function(specialized, type_reflection)


def validate_specialize(context: BindContext, signature: BoundCall, function: SlangFunction):
    # Get sorted list of root parameters for trampoline function
    root_params = [
        y
        for y in sorted(
            signature.args + list(signature.kwargs.values()),
            key=lambda x: x.param_index,
        )
        if y.param_index >= 0 and y.param_index < len(function.parameters)
    ]

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
                f"After implicit casting, cannot convert {input} to TypeReflection."
            )

    types = [to_type_reflection(x.vector_type) for x in root_params]
    for type, param in zip(types, root_params):
        if type is None:
            raise KernelGenException(
                f"After implicit casting, unable to find reflection data for {param.variable_name}"
                "This typically suggests the binding system has attempted to generate an invalid Slang type."
            )

    specialized = function.reflection.specialize_with_arg_types(types)
    if specialized is None:
        raise KernelGenException(
            "After implicit casting, no Slang overload found that matches the provided Python argument types. "
            "This typically suggests SlangPy selected an overload to call, but couldn't find a valid "
            "way to pass your Python arguments to it."
        )


def bind(context: BindContext, signature: BoundCall, function: SlangFunction) -> BoundCall:
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
            b.bind(function.return_type, {ModifierID.out}, "_result")
        elif x.param_index == -1:
            assert function.this is not None
            b.bind(
                function.this,
                {ModifierID.inout if function.mutating else ModifierID.inn},
                "_this",
            )
        else:
            b.bind(function.parameters[x.param_index])

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
            b.bind(function.parameters[v.param_index])

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


def generate_constants(build_info: "FunctionBuildInfo", cg: CodeGen):
    if build_info.constants is not None:
        for k, v in build_info.constants.items():
            if isinstance(v, bool):
                cg.constants.append_statement(
                    f"export static const bool {k} = {'true' if v else 'false'}"
                )
            elif isinstance(v, (int, float)):
                cg.constants.append_statement(f"export static const {type(v).__name__} {k} = {v}")
            else:
                raise KernelGenException(
                    f"Constant value '{k}' must be an int, float or bool, not {type(v).__name__}"
                )


def generate_code(
    context: BindContext,
    build_info: "FunctionBuildInfo",
    signature: BoundCall,
    cg: CodeGen,
):
    """
    Generate a list of call data nodes that will be used to generate the call
    """
    nodes: list[BoundVariable] = []

    # Generate the header
    cg.add_import("slangpy")

    call_data_len = context.call_dimensionality

    # Get the call group size so we can see about using it when generating the
    # [numthreads(...)] attribute. We use 1 as the default size if a call
    # group shape has not been set, as we can use that to make things "linear".
    # Note that when size is 1, we will still launch a group of 32 threads,
    # but each thread is conceptually in its own group of size 1. This then
    # leads to a linearly calculated call_id based on threadID.x and the call
    # shape's size and strides.
    call_group_size = 1
    call_group_shape = build_info.call_group_shape
    if call_group_shape is not None:
        call_group_shape_vector = call_group_shape.as_list()

        # Validate call_group_shape dimensions and values before using them
        if len(call_group_shape_vector) > context.call_dimensionality:
            raise KernelGenException(
                f"call_group_shape dimension ({len(call_group_shape_vector)}) must be <= "
                f"call_shape dimension ({context.call_dimensionality}). "
                f"call_group_shape cannot have more dimensions than call_shape."
            )

        # Validate that all call_group_shape values are >= 1
        for i, dim in enumerate(call_group_shape_vector):
            if dim < 1:
                raise KernelGenException(
                    f"call_group_shape[{i}] = {dim} is invalid. "
                    f"All call_group_shape elements must be >= 1."
                )

        # Calculate call group size as product of all dimensions
        for dim in call_group_shape_vector:
            call_group_size *= dim
        # Check if call_group_size exceeds hardware limits
        if call_group_size > 1024:
            raise KernelGenException(
                f"call_group_size ({call_group_size}) exceeds the typical 1024 maximum "
                f"enforced by most APIs. Consider reducing your call_group_shape dimensions."
            )

    # Specify some global-scope static variables here to allow users to make calls
    # to functions like get_call_id<N>() etc in order to access call_id,
    # call_group_id, and call_group_thread_id.

    # Note: Support for static global variables should be seen as a legacy feature
    #       and its use is discouraged. As such, this may need to be reworked in
    #       the future.
    if call_data_len > 0:
        # Context wants call_id as an int for some reason, so respect that here
        # for now
        cg.add_snippet("call_id", f"static int[{call_data_len}] call_id;\n")
        cg.add_snippet("call_group_id", f"static uint[{call_data_len}] call_group_id;\n")
        cg.add_snippet(
            "call_group_thread_id", f"static uint[{call_data_len}] call_group_thread_id;\n"
        )

        # TODO: Ideally there would be a way to just return call_id here so we could avoid wasted effort itterating.
        #       Not enough of a slang expert at the moment to know what that could/should look like.
        cg.add_snippet(
            "get_call_id",
            f"export public static int[N] get_call_id<let N: int>() {{ int[N] ret;  for(int i=0; i<{call_data_len}; i++) {{ ret[i] = call_id[i]; }} return ret; }};\n",
        )
        cg.add_snippet(
            "get_call_group_id",
            f"export public static uint[N] get_call_group_id<let N: int>() {{ uint[N] ret;  for(int i=0; i<{call_data_len}; i++) {{ ret[i] = call_group_id[i]; }} return ret; }};\n",
        )
        cg.add_snippet(
            "get_call_group_thread_id",
            f"export public static uint[N] get_call_group_thread_id<let N: int>() {{ uint[N] ret;  for(int i=0; i<{call_data_len}; i++) {{ ret[i] = call_group_thread_id[i]; }} return ret; }};\n",
        )

    cg.add_import(build_info.module.name)

    # Generate constants if specified
    generate_constants(build_info, cg)

    # Generate call data inputs if vector call
    if call_data_len > 0:
        # We use the call shape dimensions to detect cases when the call shape
        # and the call group shape are not aligned. When a thread's call id
        # falls outside the call shape, we need it to return early. This is
        # similar to the default linear case when the call shape size is not
        # 32 thread aligned.
        cg.call_data.append_statement(f"int[{call_data_len}] _call_dim")
        # A group can be thought of as a "window" looking at a
        # portion of the entire call shape. Grid here refers to the
        # N dimensional call shape being broken up into some number of N
        # dimensional "window"s / groups.
        cg.call_data.append_statement(f"uint[{call_data_len}] _grid_stride")
        cg.call_data.append_statement(f"uint[{call_data_len}] _grid_dim")
        cg.call_data.append_statement(f"uint[{call_data_len}] _group_stride")
        cg.call_data.append_statement(f"uint[{call_data_len}] _group_dim")

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
            data_name = (
                f"_param_{x.variable_name}" if x.create_param_block else f"data.{x.variable_name}"
            )
            cg.trampoline.append_statement(
                f"{data_name}.load(context.map(_m_{x.variable_name}), {x.variable_name})"
            )

    cg.trampoline.append_indent()
    if any(x.variable_name == "_result" for x in root_params):
        cg.trampoline.append_code(f"_result = ")

    # Get function name, if it's the init function, use the result type
    func_name = build_info.name
    if func_name == "$init":
        results = [x for x in root_params if x.variable_name == "_result"]
        assert len(results) == 1
        assert results[0].vector_type is not None
        func_name = results[0].vector_type.full_name
    elif len(root_params) > 0 and root_params[0].variable_name == "_this":
        func_name = f"_this.{func_name}"

    # Get the parameters that are not the result or this reference
    normal_params = [
        x for x in root_params if x.variable_name != "_result" and x.variable_name != "_this"
    ]

    # Internal call to the actual function
    cg.trampoline.append_code(
        f"{func_name}(" + ", ".join(x.variable_name for x in normal_params) + ");\n"
    )

    # For each writable trampoline parameter, potentially store it
    for x in root_params:
        if (
            x.access[0] == AccessType.write
            or x.access[0] == AccessType.readwrite
            or x.access[1] == AccessType.read
        ):
            if not x.python.is_writable:
                raise BoundVariableException(f"Cannot read back value for non-writable type", x)
            data_name = (
                f"_param_{x.variable_name}" if x.create_param_block else f"data.{x.variable_name}"
            )
            cg.trampoline.append_statement(
                f"{data_name}.store(context.map(_m_{x.variable_name}), {x.variable_name})"
            )

    cg.trampoline.end_block()
    cg.trampoline.append_line("")

    # Generate the main function
    cg.kernel.append_line('[shader("compute")]')
    if call_group_size != 1:
        cg.kernel.append_line(f"[numthreads({call_group_size}, 1, 1)]")
    else:
        cg.kernel.append_line("[numthreads(32, 1, 1)]")

    # Note: While flat_call_thread_id is 3-dimensional, we consider it "flat" and 1-dimensional because of the
    #       true call group shape of [x, 1, 1] and only use the first dimension for the call thread id.
    cg.kernel.append_line("void compute_main(uint3 flat_call_thread_id: SV_DispatchThreadID)")
    cg.kernel.begin_block()
    cg.kernel.append_statement("if (any(flat_call_thread_id >= call_data._thread_count)) return")

    # Calculate the ids that we'll use to compute the call id. We'll also store the
    # N-dimensional call_*_id's as global-scope static variables such that they can
    # be "queried" in user shaders via getters provided by the slangpy module.
    if call_data_len > 0:
        # Calculate flattened id's that will be used to determine the N-dimensional
        # id's.
        cg.kernel.append_statement(f"uint flat_call_id = flat_call_thread_id.x")
        # We use call_group_size here, which can be 1, because we want to allow for
        # a default linear dispatch when the call group shape is not specified. When
        # call_group_size is 1, each thread conceptually has its own group (even though
        # they will stil be dispatched in groups of 32), where the call_group_id will match
        # the call_id. This results in linear mapping from the flat thread id to the
        # call id where call_id[i] = flat_call_id/call_stride[i] % call_dim[i].
        cg.kernel.append_statement(
            f"uint flat_call_group_id = flat_call_thread_id.x / {call_group_size}"
        )
        # Again, we want to use flat_call_thread_id.x for calcualting flat_call_group_thread_id
        # rather than something like SV_GroupIndex because we want to be able to force
        # flat_call_group_thread_id == flat_call_thread_id.x in the default case using
        # linear dispatches. The same could be accomplished with SV_GroupIndex and
        # [numthreads(1, 1, 1)], but this would likely be less performant.
        cg.kernel.append_statement(
            f"uint flat_call_group_thread_id = flat_call_thread_id.x % {call_group_size}"
        )

    # Loads / initializes call id
    context_args = "flat_call_thread_id"
    if call_data_len > 0:
        cg.kernel.append_line(f"for (int i=0; i<{call_data_len}; i++)")
        cg.kernel.append_line("{")
        cg.kernel.inc_indent()

        # Calculate the N-dimensional id's
        cg.kernel.append_statement(
            "call_group_thread_id[i] = (flat_call_group_thread_id/call_data._group_stride[i]) % call_data._group_dim[i]"
        )
        cg.kernel.append_statement(
            "call_group_id[i] = (flat_call_group_id/call_data._grid_stride[i]) % call_data._grid_dim[i]"
        )
        cg.kernel.append_statement(
            f"call_id[i] = call_group_id[i] * call_data._group_dim[i] + call_group_thread_id[i]"
        )
        # In the event that the call shape is not aligned to the call group shape,
        # we use an aligned call shape to calculate the number of threads that we
        # need. However, that means that some threads will fall outside the call shape
        # and as a result will need to return early and be wasted.
        cg.kernel.append_statement("if (call_id[i] >= call_data._call_dim[i]) return")
        cg.kernel.dec_indent()
        cg.kernel.append_statement("}")

        context_args += f", call_id"

    cg.kernel.append_statement(f"Context context = {{{context_args}}}")

    # Call the trampoline function
    fn = trampoline_fn
    if context.call_mode == CallMode.bwds:
        fn = f"bwd_diff({fn})"
    cg.kernel.append_statement(f"{fn}(context, call_data)")

    cg.kernel.end_block()
