import hashlib
from io import StringIO
from typing import Any, Optional, Union, cast
from kernelfunctions.backend import Device
from kernelfunctions.codegen import CodeGen
from kernelfunctions.function import Function
from kernelfunctions.shapes import TConcreteShape
from kernelfunctions.types.boundvariable import BoundCall, BoundVariable
from kernelfunctions.typeregistry import PYTHON_SIGNATURE_HASH
from kernelfunctions.types import CallMode, AccessType
from kernelfunctions.types.buffertype import NDDifferentiableBufferType
from kernelfunctions.types.enums import IOType
from kernelfunctions.types.pythonvalue import PythonFunctionCall, PythonVariable
from kernelfunctions.types.slangvalue import SlangFunction, SlangVariable
from kernelfunctions.types.valuereftype import ValueRefType


def build_signature_hash(*args: Any, **kwargs: Any) -> str:
    """
    Build a sha256 hash that uniquely identifies a function call signature
    """
    stream = StringIO()
    stream.write("args\n")
    for x in args:
        stream.write("-\n")
        _recurse_build_signature_hash(stream, x)
    stream.write("kwargs\n")
    for x, y in sorted(kwargs.items()):
        stream.write(x)
        stream.write("\n")
        _recurse_build_signature_hash(stream, y)
    return hashlib.sha256(stream.getvalue().encode()).hexdigest()


def _recurse_build_signature_hash(stream: StringIO, python_value: Any):
    """
    Internal recursive function to populate a string IO buffer
    that'll be used to generate sha256 hash
    """

    val_type = type(python_value)
    stream.write(val_type.__name__)
    stream.write("\n")

    hasher = PYTHON_SIGNATURE_HASH.get(val_type, None)
    if hasher is not None:
        hasher(stream, python_value)
        stream.write("\n")

    if isinstance(python_value, dict):
        for key, value in sorted(python_value.items()):
            stream.write(key)
            stream.write("\n")
            _recurse_build_signature_hash(stream, value)


def build_signature(*args: Any, **kwargs: Any):
    """
    Builds a basic call signature for a given set of python 
    arguments and keyword arguments
    """
    # arg_signatures = [SignatureNode(x) for x in args]
    # kwarg_signatures = {k: SignatureNode(v) for k, v in kwargs.items()}
    # return (arg_signatures, kwarg_signatures)
    return PythonFunctionCall(*args, **kwargs)


def match_signatures(
    signature: PythonFunctionCall,
    function: SlangFunction,
    call_mode: CallMode
) -> Union[None, dict[PythonVariable, SlangVariable]]:
    """
    Attempts to efficiently match a signature to a slang function overload.
    Returns a dictionary of matched argument nodes to parameter names.
    """

    # Bail instantly if trying to call a non-differentiable function with a derivative call
    if call_mode != CallMode.prim and not function.differentiable:
        return None

    overload_parameters = function.parameters

    args = signature.args.copy()
    kwargs = signature.kwargs.copy()
    rval = None
    matched_rval: Optional[BoundVariable] = None

    # Check for user providing return value. In all cases, it
    # can be explicitly passed as a keyword argument. With derivative
    # calls, if not provided as keyword, it is EXPECTED to be
    # the last positional argument.
    if "_result" in kwargs:
        rval = kwargs["_result"]
        del kwargs["_result"]
        if function.return_value is None:
            raise ValueError(
                f"Function {function.name} does not return a value, but one was provided")
        if not rval.is_compatible(function.return_value):
            return None
    elif call_mode != CallMode.prim and function.return_value is not None:
        rval = args[-1]
        args = args[:-1]
        if not rval.is_compatible(function.return_value):
            return None

    # If there are more positional arguments than parameters, it can't match.
    if len(args) > len(overload_parameters):
        return None

    # Dictionary of slang arguments and corresponding python arguments
    param_name_to_index = {x.name: i for i, x in enumerate(overload_parameters)}
    matched_params: dict[SlangVariable, Optional[PythonVariable]] = {
        x: None for x in overload_parameters}

    # Positional arguments must all match perfectly
    for i, arg in enumerate(args):
        param = overload_parameters[i]
        if not arg.is_compatible(param):
            return None
        arg.param_index = i
        matched_params[param] = arg

    # Pair up kw arguments with slang arguments
    for name, arg in kwargs.items():
        i = param_name_to_index.get(name)
        if i is None:
            return None
        param = overload_parameters[i]
        if not arg.is_compatible(param):
            return None
        arg.param_index = i
        matched_params[param] = arg

    # Check if all arguments have been handled
    for param in matched_params.values():
        if param is None:
            return None

    if rval is not None:
        assert function.return_value is not None
        matched_params[function.return_value] = rval

    inverse_match = {
        v: k for k, v in matched_params.items()}

    return inverse_match  # type: ignore


def bind(
        signature: PythonFunctionCall,
        mapping: dict[PythonVariable, SlangVariable],
        call_mode: CallMode,
        input_transforms: Optional[dict[str, TConcreteShape]] = None,
        output_transforms: Optional[dict[str, TConcreteShape]] = None) -> BoundCall:
    """
    Apply a matched signature to a slang function, adding slang type marshalls
    to the signature nodes and performing other work that kicks in once
    match has occured.
    """
    res = BoundCall()
    res.args = [BoundVariable(x, mapping[x], call_mode, input_transforms,
                              output_transforms) for x in signature.args]
    res.kwargs = {k: BoundVariable(
        v, mapping[v], call_mode, input_transforms, output_transforms) for k, v in signature.kwargs.items()}
    return res


def calculate_and_apply_call_shape(signature: BoundCall) -> list[int]:
    """
    Given the shapes of the parameters (inferred from slang) and inputs (passed in by the user), 
    calculates the argument shapes and call shape.
    All parameters must have a shape, however individual dimensions can have an undefined size (None).
    Inputs can also be fully undefined.
    """

    # Get all arguments that are to be written
    nodes: list[BoundVariable] = []
    for node in signature.values():
        node.get_input_list(nodes)

    # Find the highest dimension in the mappings. Note: for a purely scalar
    # call, highest dimensionality can be 0, so we start at -1.
    highest_output_dimensionality = -1
    for node in nodes:
        if node.call_transform is not None:
            for i in node.call_transform:
                highest_output_dimensionality = max(highest_output_dimensionality, i)

    # Call shape has the number of dimensions that the largest argument has
    call_shape: list[Optional[int]] = [
        None for _ in range(highest_output_dimensionality + 1)
    ]

    # Numpy rules for calculating broadcast dimension sizes, with additional
    # rules for handling undefined dimensions
    for node in nodes:
        if node.argument_shape is not None:
            assert node.call_transform is not None
            for arg_dim_idx in range(len(node.argument_shape)):
                call_dim_idx = node.call_transform[arg_dim_idx]
                arg_dim_size = node.argument_shape[arg_dim_idx]
                call_dim_size = call_shape[call_dim_idx]
                if call_dim_size is None:
                    call_dim_size = arg_dim_size
                elif call_dim_size == 1:
                    call_dim_size = arg_dim_size
                elif arg_dim_size == 1:
                    pass  # call dim already set and arg dim is 1 so can be broadcast
                elif arg_dim_size is not None and call_dim_size != arg_dim_size:
                    raise ValueError(
                        f"Arg {node.param_index}, CS[{call_dim_idx}] != AS[{arg_dim_idx}], {call_dim_size} != {arg_dim_size}"
                    )
                call_shape[call_dim_idx] = call_dim_size

    # Assign the call shape to any fully undefined argument shapes
    for node in nodes:
        if node.argument_shape is None:
            node.argument_shape = call_shape
            node.call_transform = [i for i in range(len(call_shape))]

    # Raise an error if the call shape is still undefined
    if None in call_shape:
        raise ValueError(f"Call shape is ambiguous: {call_shape}")
    verified_call_shape = cast(list[int], call_shape)

    # Populate any still-undefined argument shapes from the call shape
    for node in nodes:
        assert node.argument_shape is not None
        assert node.call_transform is not None
        for arg_dim_idx in range(len(node.argument_shape)):
            call_dim_idx = node.call_transform[arg_dim_idx]
            if node.argument_shape[arg_dim_idx] is None:
                node.argument_shape[arg_dim_idx] = verified_call_shape[call_dim_idx]
        if None in node.argument_shape:
            raise ValueError(
                f"Arg {node.param_index} shape is ambiguous: {node.argument_shape}")

    # Populate the actual input transforms to be used in kernel
    for node in nodes:
        assert node.argument_shape is not None
        assert node.call_transform is not None
        if node.python.container_shape is not None:
            node.loadstore_transform = [
                None for x in range(len(node.python.container_shape))]
            for i in range(len(node.python.container_shape)):
                arg_dim_idx = i
                if node.transform_inputs is not None:
                    arg_dim_idx = node.transform_inputs[i]
                if arg_dim_idx < len(node.call_transform):
                    cont_shape = node.python.container_shape[arg_dim_idx]
                    if cont_shape is not None and cont_shape > 1:
                        node.loadstore_transform[i] = node.call_transform[arg_dim_idx]
        else:
            node.loadstore_transform = []

    return verified_call_shape


def create_return_value(call_shape: list[int], signature: BoundCall, mode: CallMode):
    """
    Create the return value for the call
    """
    if mode == CallMode.prim:
        node = signature.kwargs.get("_result")
        if node is not None and node.python.primal_type_name == 'none':
            node.argument_shape = call_shape  # type: ignore (valid)
            node.call_transform = [i for i in range(len(call_shape))]
            node.loadstore_transform = [i for i in range(len(call_shape))]
            if len(call_shape) == 0:
                node.python.set_type(ValueRefType(node.slang.primal))
            else:
                node.python.set_type(NDDifferentiableBufferType(node.slang.primal))


def generate_code(call_shape: list[int], function: Function, signature: BoundCall, mode: CallMode, cg: CodeGen):
    """
    Generate a list of call data nodes that will be used to generate the call
    """
    nodes: list[BoundVariable] = []

    # Generate the header
    cg.add_import("slangpy")
    cg.add_import(function.module.name)

    # Generate call data inputs if vector call
    call_data_len = len(call_shape)
    if call_data_len > 0:
        cg.call_data.append_statement(f"int[{call_data_len}] _call_stride")
        cg.call_data.append_statement(f"int[{call_data_len}] _call_dim")
    cg.call_data.append_statement(f"uint3 _thread_count")

    # Generate the context structure
    cg.context.append_line(f"struct Context: IContext")
    cg.context.begin_block()
    cg.context.append_statement(f"uint3 thread_id")
    cg.context.append_statement(f"int[{max(1,call_data_len)}] call_id")
    cg.context.append_line("uint3 get_thread_id() { return thread_id; }")
    cg.context.append_line("int get_call_id(int dim) { return call_id[dim]; }")

    cg.context.end_block()

    # Generate call data definitions for all inputs to the kernel
    for node in signature.values():
        node.gen_call_data_code(cg)

    # Get sorted list of root parameters for trampoline function
    root_params = sorted(signature.values(), key=lambda x: x.param_index)

    # Generate the trampoline function
    root_param_defs = [x._gen_trampoline_argument() for x in root_params]
    root_param_defs = ", ".join(root_param_defs)
    cg.trampoline.append_line("[Differentiable]")
    cg.trampoline.append_line("void _trampoline(" + root_param_defs + ")")
    cg.trampoline.begin_block()
    cg.trampoline.append_indent()
    if any(x.path is '_result' for x in root_params):
        cg.trampoline.append_code(f"_result = ")
    cg.trampoline.append_code(
        f"{function.name}(" + ", ".join(x.slang.name for x in root_params if x.slang.name != '_result') + ");\n")
    cg.trampoline.end_block()
    cg.trampoline.append_line("")

    # Generate the main function
    cg.kernel.append_line('[shader("compute")]')
    cg.kernel.append_line("[numthreads(32, 1, 1)]")
    cg.kernel.append_line("void main(uint3 dispatchThreadID: SV_DispatchThreadID)")
    cg.kernel.begin_block()
    cg.kernel.append_statement(
        "if (any(dispatchThreadID >= call_data._thread_count)) return")

    # Loads / initializes call id (inserting dummy if not vector call)
    cg.kernel.append_statement("Context context")
    cg.kernel.append_statement("context.thread_id = dispatchThreadID")
    if call_data_len > 0:
        for i in range(call_data_len):
            cg.kernel.append_statement(
                f"context.call_id[{i}] = (dispatchThreadID.x/call_data._call_stride[{i}]) % call_data._call_dim[{i}]")
    else:
        cg.kernel.append_statement("context.call_id = {0}")

    def declare_p(x: BoundVariable, has_suffix: bool = False):
        name = f"{x.variable_name}{'_p' if has_suffix else ''}"
        cg.kernel.append_statement(f"_{x.variable_name}::primal_type {name}")
        return name

    def declare_d(x: BoundVariable, has_suffix: bool = False):
        assert x.slang.derivative is not None
        name = f"{x.variable_name}{'_d' if has_suffix else ''}"
        cg.kernel.append_statement(f"_{x.variable_name}::derivative_type {name}")
        return name

    def load_p(x: BoundVariable, has_suffix: bool = False):
        n = declare_p(x, has_suffix)
        cg.kernel.append_statement(
            f"call_data.{x.variable_name}.load_primal(context,{n})")
        return n

    def load_d(x: BoundVariable, has_suffix: bool = False):
        n = declare_d(x, has_suffix)
        cg.kernel.append_statement(
            f"call_data.{x.variable_name}.load_derivative(context,{n})")
        return n

    def store_p(x: BoundVariable, has_suffix: bool = False):
        cg.kernel.append_statement(
            f"call_data.{x.variable_name}.store_primal(context,{x.variable_name}{'_p' if has_suffix else ''})")

    def store_d(x: BoundVariable, has_suffix: bool = False):
        cg.kernel.append_statement(
            f"call_data.{x.variable_name}.store_derivative(context,{x.variable_name}{'_d' if has_suffix else ''})")

    def create_pair(x: BoundVariable, inc_derivative: bool):
        p = load_p(x, True)
        if not inc_derivative:
            cg.kernel.append_statement(
                f"var {x.variable_name} = diffPair({p})")
        else:
            d = load_d(x, True)
            cg.kernel.append_statement(
                f"var {x.variable_name} = diffPair({p}, {d})")

    def store_pair_derivative(x: BoundVariable):
        cg.kernel.append_statement(
            f"call_data.{x.variable_name}.store_derivative(context,{x.variable_name}.d)")

    # Select either primals, derivatives or pairs for the trampoline function
    names: list[str] = []
    for x in root_params:
        (primal_access, derivative_access) = x.access
        if primal_access != AccessType.none and derivative_access != AccessType.none:
            assert not primal_access in [AccessType.write, AccessType.readwrite]
            assert derivative_access in [AccessType.write, AccessType.readwrite]
            create_pair(x, derivative_access == AccessType.readwrite)
        else:
            if primal_access == AccessType.read or primal_access == AccessType.readwrite:
                load_p(x)
            elif primal_access == AccessType.write:
                declare_p(x)
            if derivative_access == AccessType.read or derivative_access == AccessType.readwrite:
                load_d(x)
            elif derivative_access == AccessType.write:
                declare_d(x)
        if primal_access != AccessType.none or derivative_access != AccessType.none:
            names.append(f"{x.variable_name}")

    # Call the trampoline function
    fn = "_trampoline"
    if mode == CallMode.bwds:
        fn = f"bwd_diff({fn})"
    cg.kernel.append_statement(
        f"{fn}(" + ", ".join(names) + ")")

    # For each writable trampoline parameter, potentially store it
    for x in root_params:
        (primal_access, derivative_access) = x.access
        if primal_access != AccessType.none and derivative_access != AccessType.none:
            store_pair_derivative(x)
        else:
            if primal_access == AccessType.write or primal_access == AccessType.readwrite:
                store_p(x)
            if derivative_access == AccessType.write or derivative_access == AccessType.readwrite:
                store_d(x)

    cg.kernel.end_block()


def write_calldata_pre_dispatch(device: Device, call_signature: BoundCall, call_data: dict[str, Any], *args: Any, **kwargs: Any):
    """
    Write the call data for args + kwargs before dispatching
    """
    sig_args = call_signature.args
    sig_kwargs = call_signature.kwargs

    for idx, value in enumerate(args):
        sig_args[idx].write_call_data_pre_dispatch(device, call_data, value)

    for key, value in kwargs.items():
        sig_kwargs[key].write_call_data_pre_dispatch(device, call_data, value)


def read_call_data_post_dispatch(device: Device, call_signature: BoundCall, call_data: dict[str, Any], *args: Any, **kwargs: Any):
    """
    Read the call data for args + kwargs after dispatching
    """
    sig_args = call_signature.args
    sig_kwargs = call_signature.kwargs

    for idx, value in enumerate(args):
        sig_args[idx].read_call_data_post_dispatch(device, call_data, value)
    for key, value in kwargs.items():
        sig_kwargs[key].read_call_data_post_dispatch(device, call_data, value)


def get_readable_signature_string(call_signature: PythonFunctionCall):
    text: list[str] = []
    for idx, arg in enumerate(call_signature.args):
        text.append(f"arg{idx}: ")
        text.append(arg._recurse_str(1))
        text.append("\n")
    for key, arg in call_signature.kwargs.items():
        text.append(f"{key}: ")
        text.append(arg._recurse_str(1))
        text.append("\n")
    return "".join(text)


def get_readable_func_string(slang_function: Optional[SlangFunction]):
    if slang_function is None:
        return ""

    def get_modifiers(val: SlangVariable):
        mods: list[str] = []
        if val.io_type == IOType.inout:
            mods.append("inout")
        elif val.io_type == IOType.out:
            mods.append("out")
        if val.no_diff:
            mods.append("nodiff")
        return " ".join(mods)

    text: list[str] = []
    if slang_function.return_value is not None:
        text.append(f"{slang_function.return_value.primal_type_name} ")
    else:
        text.append("void ")
    text.append(slang_function.name)
    parms = [
        f"{get_modifiers(x)}{x.primal_type_name} {x.name}" for x in slang_function.parameters]
    text.append(", ".join(parms))
    text.append(")")
    return "".join(text)
