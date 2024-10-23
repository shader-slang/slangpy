from typing import TYPE_CHECKING, Any, Optional, Union

from kernelfunctions.core import (
    CodeGen,
    IOType, CallMode, AccessType,
    BindContext, ReturnContext, BoundCall, BoundVariable, BoundVariableException,
    SlangFunction, SlangVariable,
    PythonFunctionCall, PythonVariable,
    BoundCallRuntime, BoundVariableRuntime,
    Shape
)

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


def match_signatures(
    signature: PythonFunctionCall,
    function: SlangFunction,
    call_mode: CallMode
) -> Union[MismatchReason, dict[PythonVariable, SlangVariable]]:
    """
    Attempts to efficiently match a signature to a slang function overload.
    Returns a dictionary of matched argument nodes to parameter names.
    """

    # Bail instantly if trying to call a non-differentiable function with a derivative call
    if call_mode != CallMode.prim and not function.differentiable:
        return MismatchReason("Function is not differentiable")

    # Easy step: Match arguments to parameters based on position/names
    # Bail if too many/few args, mismatched parameter name, multiple asignments, etc.
    overload_parameters = function.parameters.copy()
    if function.return_value is not None:
        overload_parameters.append(function.return_value)

    if len(signature.args) > len(overload_parameters):
        return MismatchReason(f"Too many function arguments: Expected {len(function.parameters)}, "
                              f"received {len(signature.args) + len(signature.kwargs)}")

    # Build empty positional list of python arguments to correspond to each slang argument
    positioned_args: list[Optional[PythonVariable]] = [None] * len(overload_parameters)

    # Populate the first N arguments from provided positional arguments
    for i, arg in enumerate(signature.args):
        positioned_args[i] = arg

    # Attempt to populate the remaining arguments from keyword arguments
    name_map = {param.name: i for i, param in enumerate(overload_parameters)}
    for name, arg in signature.kwargs.items():
        if name not in name_map:
            return MismatchReason(f"No parameter named '{name}'")
        i = name_map[name]
        if positioned_args[i] is not None:
            return MismatchReason(f"Parameter '{name}' is already assigned")
        positioned_args[i] = arg

    # Identify missing required params (TODO: This really needs to understand
    # how default values must be populated from left-to-right for a valid call)
    missing_params = [param for arg, param in zip(positioned_args, overload_parameters)
                      if arg is None and not param.has_default]
    if missing_params:
        missing_names = "', '".join(param.name for param in missing_params)
        return MismatchReason(f"Arguments missing for parameter(s) '{missing_names}'")

    # Build dictionary of matched arguments
    matched_args = {arg: param for arg, param in zip(positioned_args, overload_parameters)
                    if arg is not None}

    # Each parameter without default is matched to exactly one argument.
    # Now check if the types are compatible

#    for arg, param in matched_args.items():
#        if not arg.is_compatible(param):
#            return MismatchReason(f"Cannot convert from {arg.primal.name} to "
#                                  f"{param.primal.name} for parameter '{param.name}'")
#
#        specialized = param.specialize(arg)
#        if specialized is None:
#            return MismatchReason(f"Could not specialize {param.primal.name} to {arg.primal.name} for parameter '{param.name}'")
#        matched_args[arg] = specialized

    return matched_args


def bind(
        context: BindContext,
        signature: PythonFunctionCall,
        mapping: dict[PythonVariable, SlangVariable],
        output_transforms: Optional[dict[str, 'TShapeOrTuple']] = None) -> BoundCall:
    """
    Apply a matched signature to a slang function, adding slang type marshalls
    to the signature nodes and performing other work that kicks in once
    match has occured.
    """

    # First bind things
    res = BoundCall()
    res.args = [BoundVariable(x, mapping[x],
                              output_transforms) for x in signature.args]
    res.kwargs = {k: BoundVariable(
        v, mapping[v], output_transforms) for k, v in signature.kwargs.items()}

    return res


def apply_vectorization(context: BindContext, call: BoundCall):
    return call


def apply_bindings(context: BindContext, call: BoundCall):
    """
    Recursively step through all parameters in the bind call and generate
    any data that requires both PythonVariable and SlangVariable to be
    fully resolved.
    """
    try:
        for arg in call.args:
            arg.apply_binding(context)
        for arg in call.kwargs.values():
            arg.apply_binding(context)
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
        return str(variable.slang.primal.get_shape().as_list())
    else:
        return "None"


def _gen_python_shape_string(variable: BoundVariable) -> str:
    if variable.python.dimensionality is not None:
        return str([None]*variable.python.dimensionality)
    else:
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
            text.append(clip_string(variable.python.primal_type_name, width))
        elif name == "Output Type":
            text.append(clip_string(variable.slang.primal_type_name, width))
        elif name == "Input Shape":
            text.append(clip_string(_gen_python_shape_string(variable), width))
        elif name == "Argument Shape":
            text.append(clip_string(_gen_arg_shape_string(variable), width))
        elif name == "Type Shape":
            text.append(clip_string(_gen_type_shape_string(variable), width))
        elif name == "Transform":
            text.append(clip_string(variable.transform, width))
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


def finalize_transforms(context: BindContext, signature: BoundCall):
    try:
        nodes: list[BoundVariable] = []
        for node in signature.values():
            node.get_input_list(nodes)
        for input in nodes:
            if input.call_dimensionality is None:
                raise BoundVariableException(
                    "Unresolved call dimensionality for argument", input)
            assert input.transform.valid
            input.transform = Shape(tuple([input.transform[i] if input.transform[i] >= 0 else i +
                                           context.call_dimensionality - input.call_dimensionality for i in range(0, len(input.transform))]))
    except BoundVariableException as e:
        raise ValueError(generate_call_shape_error_string(
            signature, [], e.message, e.variable))


def create_return_value_binding(context: BindContext, signature: BoundCall, return_type: Any):
    """
    Create the return value for the call
    """

    # If return values are not needed or already set, early out
    if context.call_mode != CallMode.prim:
        return
    node = signature.kwargs.get("_result")
    if node is None or node.python.primal_type_name != 'none':
        return

    # If no desired return type was specified explicitly, fill in a useful default
    if return_type is None:
        if context.call_dimensionality == 0:
            return_type = ValueRef
        elif node.slang.primal.differentiable:
            return_type = NDDifferentiableBuffer
        else:
            return_type = NDBuffer

    return_ctx = ReturnContext(node.slang.primal, context)
    python_type = tr.get_or_create_type(return_type, return_ctx)

    node.call_dimensionality = context.call_dimensionality
    node.transform = Shape(tuple([i for i in range(len(python_type.get_shape()))]))
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
        func_name = results[0].slang.primal_element_name
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
        cg.kernel.append_statement(f"{x.slang.primal_type_name} {name}")
        return name

    def declare_d(x: BoundVariable, has_suffix: bool = False):
        assert x.slang.derivative is not None
        name = f"{x.variable_name}{'_d' if has_suffix else ''}"
        cg.kernel.append_statement(f"{x.slang.derivative_type_name} {name}")
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
    text.append("(")
    parms = [
        f"{get_modifiers(x)}{x.primal_type_name} {x.name}" for x in slang_function.parameters]
    text.append(", ".join(parms))
    text.append(")")
    return "".join(text)
