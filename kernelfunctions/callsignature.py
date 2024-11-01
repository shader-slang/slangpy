from typing import TYPE_CHECKING, Any, Optional, Union

from sgl import FunctionReflection, ModifierID, TypeReflection, VariableReflection

from kernelfunctions.bindings.valuetype import ValueType
from kernelfunctions.core import (
    CodeGen,
    IOType, CallMode, AccessType,
    BindContext, ReturnContext, BoundCall, BoundVariable, BoundVariableException,
    SlangFunction, SlangVariable,
    PythonFunctionCall, PythonVariable,
    BoundCallRuntime, BoundVariableRuntime
)

from kernelfunctions.core.basetype import BaseType
from kernelfunctions.types.buffer import NDBuffer, NDDifferentiableBuffer
from kernelfunctions.types.valueref import ValueRef
from kernelfunctions.shapes import TShapeOrTuple
import kernelfunctions.typeregistry as tr

if TYPE_CHECKING:
    from kernelfunctions.function import Function


def build_signature(*args: Any, **kwargs: Any):
    """
    Builds a basic call signature for a given set of python 
    arguments and keyword arguments
    """
    # arg_signatures = [SignatureNode(x) for x in args]
    # kwarg_signatures = {k: SignatureNode(v) for k, v in kwargs.items()}
    # return (arg_signatures, kwarg_signatures)
    return PythonFunctionCall(*args, **kwargs)


class MismatchReason:
    def __init__(self, reason: str):
        super().__init__()
        self.reason = reason


def specialize(
    context: BindContext,
    signature: PythonFunctionCall,
    functions: list[FunctionReflection],
    type: Optional[TypeReflection] = None
):
    # Handle current slang issue where init has to be found via ast, resulting in potential multiple functions
    if len(functions) > 1:
        if functions[0].name == "$init":
            matches = [x for x in functions if len(
                x.parameters) == signature.num_function_args]
            if len(matches) != 1:
                return MismatchReason("Could not find unique $init function")
            function = matches[0]
        else:
            return MismatchReason("Multiple functions found - should only happen with $init")
    else:
        function = functions[0]

    # Expecting 'this' argument as first parameter of none-static member functions (except for $init)
    first_arg_is_this = type is not None and not function.has_modifier(
        ModifierID.static) and function.name != "$init"

    # Require '_result' argument for derivative calls, either as '_result' named parameter or last positional argument
    last_arg_is_retval = function.return_type is not None and not "_result" in signature.kwargs and context.call_mode != CallMode.prim

    # Select the positional arguments we need to match against
    signature_args = signature.args
    if first_arg_is_this:
        signature_args[0].parameter_index = -1
        signature_args = signature_args[1:]
    if last_arg_is_retval:
        signature_args[-1].parameter_index = len(function.parameters)
        signature_args = signature_args[:-1]

    if signature.num_function_kwargs > 0 or signature.has_implicit_args:
        if function.is_overloaded:
            return MismatchReason("Cannot currently specialize overloaded function with named or implicit arguments")

        function_parameters = [x for x in function.parameters]

        # Build empty positional list of python arguments to correspond to each slang argument
        positioned_args: list[Optional[PythonVariable]] = [
            None] * len(function_parameters)

        # Populate the first N arguments from provided positional arguments
        for i, arg in enumerate(signature_args):
            positioned_args[i] = arg
            arg.parameter_index = i

        # Attempt to populate the remaining arguments from keyword arguments
        name_map = {param.name: i for i, param in enumerate(function_parameters)}
        for name, arg in signature.kwargs.items():
            if name == "_result":
                continue
            if name not in name_map:
                return MismatchReason(f"No parameter named '{name}'")
            i = name_map[name]
            if positioned_args[i] is not None:
                return MismatchReason(f"Parameter '{name}' is already assigned")
            positioned_args[i] = arg
            arg.parameter_index = i

        # Ensure all parameters are assigned
        if not all(x is not None for x in positioned_args):
            return MismatchReason("To use named or implicit arguments, all parameters must be specified")

        # Choose either explicit vector type or slang type for specialization
        inputs: list[Any] = []
        for i, python_arg in enumerate(positioned_args):
            slang_param = function_parameters[i]
            assert python_arg is not None
            if python_arg.vector_type is not None:
                inputs.append(python_arg.vector_type)
            elif slang_param.type.kind != TypeReflection.Kind.none and slang_param.type.kind != TypeReflection.Kind.interface:
                inputs.append(slang_param.type)
            elif isinstance(python_arg.primal, ValueType):
                inputs.append(python_arg.primal)
            else:
                raise ValueError(
                    f"Cannot specialize function with argument {i} of unknown type")
    else:
        # If no named or implicit arguments, just use explicit vector types for specialization
        inputs: list[Any] = [x.vector_type for x in signature_args]
        for i, arg in enumerate(signature_args):
            arg.parameter_index = i

    def to_type_reflection(input: Any) -> TypeReflection:
        if isinstance(input, BaseType):
            return context.device_module.layout.find_type_by_name(input.name)
        elif isinstance(input, TypeReflection):
            return input
        elif isinstance(input, str):
            return context.device_module.layout.find_type_by_name(input)
        else:
            raise ValueError(f"Cannot convert {input} to TypeReflection")

    input_types = [to_type_reflection(x) for x in inputs]
    specialized = function.specialize_with_arg_types(input_types)
    if specialized is None:
        return MismatchReason("Could not specialize function with given argument types")

    return specialized


def validate_specialize(
    context: BindContext,
    signature: PythonFunctionCall,
    function: FunctionReflection
):
    # Get sorted list of root parameters for trampoline function
    root_params = [y for y in sorted(signature.args + list(signature.kwargs.values()), key=lambda x: x.parameter_index)
                   if y.parameter_index >= 0 and y.parameter_index < len(function.parameters)]

    def to_type_reflection(input: Any) -> TypeReflection:
        if isinstance(input, BaseType):
            return context.device_module.layout.find_type_by_name(input.name)
        elif isinstance(input, TypeReflection):
            return input
        elif isinstance(input, str):
            return context.device_module.layout.find_type_by_name(input)
        else:
            raise ValueError(f"Cannot convert {input} to TypeReflection")

    types = [to_type_reflection(x.vector_type) for x in root_params]
    if any(x is None for x in types):
        raise ValueError("Unable to resolve all types for specialization")

    specialized = function.specialize_with_arg_types(types)
    if specialized is None:
        raise ValueError("Could not specialize function with given argument types")

    return specialized


def bind(
        context: BindContext,
        signature: PythonFunctionCall,
        function: SlangFunction,
        output_transforms: Optional[dict[str, 'TShapeOrTuple']] = None) -> BoundCall:
    """
    Apply a matched signature to a slang function, adding slang type marshalls
    to the signature nodes and performing other work that kicks in once
    match has occured.
    """

    res = BoundCall()

    for x in signature.args:
        if x.parameter_index == len(function.parameters):
            assert function.return_value is not None
            res.args.append(BoundVariable(x, function.return_value, output_transforms))
        elif x.parameter_index == -1:
            assert function.this is not None
            res.args.append(BoundVariable(x, function.this, output_transforms))
        else:
            res.args.append(BoundVariable(
                x, function.parameters[x.parameter_index], output_transforms))

    for k, v in signature.kwargs.items():
        if k == "_result":
            assert function.return_value is not None
            res.kwargs[k] = BoundVariable(v, function.return_value, output_transforms)
        elif k == "_this":
            assert function.this is not None
            res.kwargs[k] = BoundVariable(v, function.this, output_transforms)
        else:
            res.kwargs[k] = BoundVariable(
                v, function.parameters[v.parameter_index], output_transforms)

    return res


def apply_explicit_vectorization(call: PythonFunctionCall, args: tuple[Any, ...], kwargs: dict[str, Any]):
    """
    Apply user supplied explicit vectorization options to the python variables.
    """
    call.apply_explicit_vectorization(args, kwargs)
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
    Recursively step through all parameters in the bind call and generate
    any data that requires both PythonVariable and SlangVariable to be
    fully resolved.
    """
    try:
        for arg in call.args:
            arg.calculate_differentiability(context)
        for arg in call.kwargs.values():
            arg.calculate_differentiability(context)
    except BoundVariableException as e:
        raise ValueError(generate_call_shape_error_string(
            call, [], e.message, e.variable))
    return call


COLS = [
    ("Idx", 6),
    ("Name", 15),
    ("Input Type", 35),
    ("Output Type", 35),
    ("Input Shape", 20),
    ("Argument Shape", 20),
    ("Type Shape", 20),
    ("Transform", 20),
]

TBoundOrRuntimeVariable = Union[BoundVariable, BoundVariableRuntime]
TBoundOrRuntimeCall = Union[BoundCall, BoundCallRuntime]


def generate_argument_info_header_columns(signature: TBoundOrRuntimeCall) -> list[str]:
    """
    Generate a header string that describes the arguments
    """
    text: list[str] = []
    for name, width in COLS:
        text.append(name)
    text.append("")  # extra column for error highlight
    return text


def clip_string(s: Any, width: int) -> str:
    width -= 2
    s = str(s)
    s = s.replace("None", "?").replace("none", "?")
    if len(s) > width:
        s = s[:width-3] + "..."
    s += '  '
    return s


def _to_var(v: TBoundOrRuntimeVariable):
    if isinstance(v, BoundVariableRuntime):
        return v._source_for_exceptions
    return v


def _gen_arg_shape_string(variable: BoundVariable) -> str:
    if variable.call_dimensionality is not None:
        return str([None]*variable.call_dimensionality)
    else:
        return "None"


def _gen_type_shape_string(variable: BoundVariable) -> str:
    if variable.slang is not None:
        return str(variable.vector_type.get_shape().as_list())
    else:
        return "None"


def _gen_python_shape_string(variable: BoundVariable) -> str:
    return "None"


def generate_argument_info_columns(variable: TBoundOrRuntimeVariable, indent: int, highlight_variable: Optional[TBoundOrRuntimeVariable] = None) -> list[str]:
    """
    Generate a string that describes the argument
    """
    variable = _to_var(variable)
    text: list[str] = []
    for name, width in COLS:
        if name == "Idx":
            text.append(clip_string(variable.param_index, width))
        elif name == "Name":
            text.append(clip_string(variable.variable_name, width))
        elif name == "Input Type":
            text.append(clip_string(variable.python.primal.name, width))
        elif name == "Output Type":
            text.append(clip_string(variable.vector_type.name, width))
        elif name == "Input Shape":
            text.append(clip_string(_gen_python_shape_string(variable), width))
        elif name == "Argument Shape":
            text.append(clip_string(_gen_arg_shape_string(variable), width))
        elif name == "Type Shape":
            text.append(clip_string(_gen_type_shape_string(variable), width))
        elif name == "Mapping":
            text.append(clip_string(variable.vector_mapping, width))
    if highlight_variable and variable == _to_var(highlight_variable):
        text.append(" <---")
    else:
        text.append("")
    return text


def generate_tree_info_table(call: TBoundOrRuntimeCall, highlight_variable: Optional[TBoundOrRuntimeVariable] = None) -> list[list[str]]:
    """
    Generate a string that describes the argument
    """
    lines: list[list[str]] = []
    lines.append(generate_argument_info_header_columns(call))
    for variable in call.args:
        assert isinstance(variable, (BoundVariable, BoundVariableRuntime))
        _generate_tree_info_table(lines, variable, 0, highlight_variable)
    for variable in call.kwargs.values():
        assert isinstance(variable, (BoundVariable, BoundVariableRuntime))
        _generate_tree_info_table(lines, variable, 0, highlight_variable)
    return lines


def _generate_tree_info_table(lines: list[list[str]], variable: TBoundOrRuntimeVariable, indent: int, highlight_variable: Optional[TBoundOrRuntimeVariable] = None):
    """
    Generate a string that describes the argument
    """
    if isinstance(variable, BoundVariableRuntime):
        variable = variable._source_for_exceptions
    lines.append(generate_argument_info_columns(variable, indent, highlight_variable))
    if variable.children is not None:
        for name, child in variable.children.items():
            _generate_tree_info_table(lines, child, indent + 1, highlight_variable)


def generate_tree_info_string(call: TBoundOrRuntimeCall, highlight_variable: Optional[TBoundOrRuntimeVariable] = None) -> str:
    table = generate_tree_info_table(call, highlight_variable)
    col_widths = [max(len(x)+2 for x in col) for col in zip(*table)]
    text: list[str] = []
    for row in table:
        text.append("".join(x.ljust(width) for x, width in zip(row, col_widths)))
    return "\n".join(text)


def generate_call_shape_error_string(signature: TBoundOrRuntimeCall, call_shape: list[int | None], message: str, highlight_variable: Optional[TBoundOrRuntimeVariable] = None) -> str:
    """
    Generate a string that describes the argument
    """
    lines: list[str] = []
    lines.append(f"Error: {message}")
    if highlight_variable is not None:
        lines.append(f"Variable: {_to_var(highlight_variable).variable_name}")
    lines.append(f"Calculated call shape: {call_shape}".replace("None", "?"))

    header = "\n".join(lines)
    header += "\n"
    header += generate_tree_info_string(signature, highlight_variable)
    return header


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
    if node is None or node.python.primal.name != 'none':
        return

    # Should have an explicit vector type by now.
    assert node.vector_type is not None

    # If no desired return type was specified explicitly, fill in a useful default
    if return_type is None:
        if context.call_dimensionality == 0:
            return_type = ValueRef
        elif node.vector_type.differentiable:
            return_type = NDDifferentiableBuffer
        else:
            return_type = NDBuffer

    return_ctx = ReturnContext(node.vector_type, context)
    python_type = tr.get_or_create_type(return_type, return_ctx)

    node.call_dimensionality = context.call_dimensionality
    node.python.set_type(python_type)


def generate_code(context: BindContext, function: 'Function', signature: BoundCall, cg: CodeGen):
    """
    Generate a list of call data nodes that will be used to generate the call
    """
    nodes: list[BoundVariable] = []

    # Generate the header
    cg.add_import("slangpy")
    cg.add_import(function.module.name)

    # Generate call data inputs if vector call
    call_data_len = context.call_dimensionality
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
        node.gen_call_data_code(cg, context)

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

    # Get function name, if it's the init function, use the result type
    func_name = function.name
    if func_name == "$init":
        results = [x for x in root_params if x.path == '_result']
        assert len(results) == 1
        func_name = results[0].vector_type.name
    elif len(root_params) > 0 and root_params[0].path == '_this':
        func_name = f'_this.{func_name}'

    # Get the parameters that are not the result or this reference
    normal_params = [x for x in root_params if x.path != '_result' and x.path != '_this']

    # Internal call to the actual function
    cg.trampoline.append_code(
        f"{func_name}(" + ", ".join(x.path for x in normal_params) + ");\n")

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
        cg.kernel.append_statement(f"{x.python.vector_type.name} {name}")
        return name

    def declare_d(x: BoundVariable, has_suffix: bool = False):
        name = f"{x.variable_name}{'_d' if has_suffix else ''}"
        cg.kernel.append_statement(f"{x.python.vector_type.name}.Differential {name}")
        return name

    def load_p(x: BoundVariable, has_suffix: bool = False):
        n = declare_p(x, has_suffix)
        cg.kernel.append_statement(
            f"call_data.{x.variable_name}.load_primal(ctx(context, _m_{x.variable_name}),{n})")
        return n

    def load_d(x: BoundVariable, has_suffix: bool = False):
        n = declare_d(x, has_suffix)
        cg.kernel.append_statement(
            f"call_data.{x.variable_name}.load_derivative(ctx(context, _m_{x.variable_name}),{n})")
        return n

    def store_p(x: BoundVariable, has_suffix: bool = False):
        cg.kernel.append_statement(
            f"call_data.{x.variable_name}.store_primal(ctx(context, _m_{x.variable_name}),{x.variable_name}{'_p' if has_suffix else ''})")

    def store_d(x: BoundVariable, has_suffix: bool = False):
        cg.kernel.append_statement(
            f"call_data.{x.variable_name}.store_derivative(ctx(context, _m_{x.variable_name}),{x.variable_name}{'_d' if has_suffix else ''})")

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
            f"call_data.{x.variable_name}.store_derivative(ctx(context, _m_{x.variable_name}),{x.variable_name}.d)")

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
    if context.call_mode == CallMode.bwds:
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


def get_readable_func_refl_string(slang_function: Optional[FunctionReflection]):
    if slang_function is None:
        return ""

    def get_modifiers(val: VariableReflection):
        mods: list[str] = []
        for m in ModifierID:
            if val.has_modifier(m):
                mods.append(m.name)
        return " ".join(mods)

    text: list[str] = []
    if slang_function.return_type is not None:
        text.append(f"{slang_function.return_type.full_name} ")
    else:
        text.append("void ")
    text.append(slang_function.name)
    text.append("(")
    parms = [
        f"{get_modifiers(x)}{x.type.full_name} {x.name}" for x in slang_function.parameters]
    text.append(", ".join(parms))
    text.append(")")
    return "".join(text)
