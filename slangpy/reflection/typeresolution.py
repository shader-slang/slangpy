# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import copy
from typing import TYPE_CHECKING, Optional, cast
from slangpy.core.native import CallMode
from slangpy.core.logging import function_reflection

if TYPE_CHECKING:
    from slangpy.reflection.reflectiontypes import SlangType, SlangFunction, SlangParameter
    from slangpy.bindings import BindContext, BoundCall, BoundVariable
    from slangpy.core.native import NativeMarshall
    from slangpy.experimental.fuse import FuseNode, FusedFunction


class ResolutionArg:
    """
    Holds a single argument for type resolution. Starts with the python marshall and slang type
    being resolved, and ends with the resolved vector type (if any). In the case of functions
    with optional parameters, the python variable can be None and the parameter type will be used
    """

    def __init__(self):
        super().__init__()
        self.binding: Optional["BoundVariable"] = None
        self.vector: Optional["SlangType"] = None
        self.slang: "SlangType" = None  # type: ignore

    @property
    def python(self) -> Optional["NativeMarshall"]:
        if self.binding:
            return self.binding.python
        return None


class ResolutionDiagnostic:
    def __init__(self):
        super().__init__()
        self.summary_lines: list[str] = []
        self.detail_lines: list[str] = []

    def summary(self, txt: str):
        self.summary_lines.append(txt)
        self.detail_lines.append(txt)

    def detail(self, txt: str):
        self.detail_lines.append(txt)

    def __str__(self):
        text = []
        text.append("Type Resolution Diagnostics:")
        text.append("")
        text.append("Summary:")
        if len(self.summary_lines) == 0:
            text.append("  <none>")
        else:
            for line in self.summary_lines:
                text.append(f"  {line}")
        text.append("")
        text.append("Details:")
        if len(self.detail_lines) == 0:
            text.append("  <none>")
        else:
            for line in self.detail_lines:
                text.append(f"  {line}")
        return "\n".join(text)


def create_arg(variable: Optional["BoundVariable"], param: "SlangParameter"):
    arg = ResolutionArg()
    arg.binding = variable
    arg.vector = variable.vector_type if variable else param.type
    arg.slang = param.type
    return arg


def clone_args(args: list[ResolutionArg]):
    new_args: list[ResolutionArg] = []
    for arg in args:
        new_arg = ResolutionArg()
        new_arg.binding = arg.binding
        new_arg.slang = arg.slang
        new_arg.vector = arg.vector
        new_args.append(new_arg)
    return new_args


def resolve_arguments(
    bind_context: "BindContext", args: list[ResolutionArg], diagnostics: ResolutionDiagnostic
):
    """
    Resolves the types of the given arguments by matching marshalls against argument
    types. Returns a list of possible resolutions, where each resolution is a list of
    ResolutionArg with the vector field filled in.
    """

    resolutions = [args]

    for i in range(len(args)):
        python, slang, vec = args[i].python, args[i].slang, args[i].vector
        if not python:
            continue

        if vec:
            types = [vec]
        else:
            if hasattr(python, "resolve_types"):
                types = python.resolve_types(bind_context, slang)
            else:
                types = [python.resolve_type(bind_context, slang)]
        if not types:
            types = []
        types = [t for t in types if t]

        if len(types) == 0:
            diagnostics.summary(
                f"  Argument {i} could not be resolved: python type {python} does not match slang type {slang.full_name}."
            )

        new_resolutions = []
        for potential_type in types:
            if not potential_type:
                continue
            for resolution in resolutions:
                new_args = clone_args(resolution)
                new_args[i].vector = cast("SlangType", potential_type)
                new_resolutions.append(new_args)
        resolutions = new_resolutions

    return resolutions


class ResolvedParam:
    def __init__(self, source_param: "SlangParameter", type: "SlangType"):
        super().__init__()
        self.source_param = source_param
        self.type = type

    @property
    def name(self) -> str:
        return self.source_param.name

    @property
    def modifiers(self):
        return self.source_param.modifiers


class ResolveResult:
    def __init__(self):
        super().__init__()
        self.args: tuple["SlangType", ...] = ()
        self.kwargs: dict[str, "SlangType"] = {}
        self.slang: tuple["SlangType", ...] = ()
        self.function: "SlangFunction" = None  # type: ignore
        self.params: list[ResolvedParam] = []


def _assign_parameter_indices(
    bind_context: "BindContext",
    function: "SlangFunction",
    bindings: "BoundCall",
    diagnostics: ResolutionDiagnostic,
    this_type: Optional["SlangType"] = None,
):
    # Log argument info
    if len(bindings.args) > 0:
        diagnostics.detail(f"  Positional python arguments:")
        for i, arg in enumerate(bindings.args):
            diagnostics.detail(f"    {i}: {arg.python}")
    if len(bindings.kwargs) > 0:
        diagnostics.detail(f"  Named python arguments:")
        for name, arg in bindings.kwargs.items():
            diagnostics.detail(f"    {name}: {arg.python}")

    assert not function.is_overloaded
    function_parameters = [x for x in function.parameters]
    signature_args = bindings.args
    signature_kwargs = bindings.kwargs

    # Expecting 'this' argument as first parameter of none-static member functions (except for constructors)
    first_arg_is_this = (
        this_type is not None and not function.static and not function.is_constructor
    )

    # Require '_result' argument for derivative calls, either as '_result' named parameter or last positional argument
    last_arg_is_retval = (
        function.return_type is not None
        and function.return_type.name != "void"
        and not "_result" in bindings.kwargs
        and bind_context.call_mode != CallMode.prim
    )

    # Select the positional arguments we need to match against
    if first_arg_is_this:
        signature_args[0].param_index = -1
        signature_args = signature_args[1:]
    if last_arg_is_retval:
        signature_args[-1].param_index = len(function.parameters)
        signature_args = signature_args[:-1]

    # Build empty positional list of python arguments to correspond to each slang argument
    positioned_args: list[Optional["BoundVariable"]] = [None] * len(function_parameters)

    # Populate the first N arguments from provided positional arguments
    if len(signature_args) > len(function_parameters):
        diagnostics.summary(
            f"  Too many arguments: expected {len(function_parameters)}, got {len(signature_args)}"
        )
        return None

    for i, arg in enumerate(signature_args):
        positioned_args[i] = arg
        arg.param_index = i

    # Attempt to populate the remaining arguments from keyword arguments
    name_map = {param.name: i for i, param in enumerate(function_parameters)}
    for name, arg in signature_kwargs.items():
        if name == "_result":
            continue
        if name not in name_map:
            diagnostics.summary(f"  No parameter named '{name}'.")
            return None
        i = name_map[name]
        if positioned_args[i] is not None:
            diagnostics.summary(f"  Parameter '{name}' specified multiple times.")
            return None
        positioned_args[i] = arg
        arg.param_index = i

    # Check all arguments resolved
    for i, arg in enumerate(positioned_args):
        if arg is None:
            diagnostics.summary(f"  Parameter '{function_parameters[i].name}' not specified.")
            return None

    # If we reach this point, all positional and keyword arguments have been matched to slang parameters
    # Return set of ResolutionArgs that contain [MarshallType, ParameterType, ResolvedType|None]
    return [create_arg(param, arg) for param, arg in zip(positioned_args, function_parameters)]


def _resolve_function_internal(
    bind_context: "BindContext",
    function: "SlangFunction",
    bindings: "BoundCall",
    diagnostics: ResolutionDiagnostic,
    this_type: Optional["SlangType"] = None,
):
    # Should never get a fused function here.
    diagnostics.summary(f"{function_reflection(function.reflection)}")

    # Map python arguments to slang parameters and get a set of ResolutionArgs back
    current_args = _assign_parameter_indices(
        bind_context, function, bindings, diagnostics, this_type
    )
    if current_args is None:
        return None

    # Use resolve_arguments to try and find 1 or more possible resolutions
    resolved_args = resolve_arguments(bind_context, current_args, diagnostics)

    # No valid resolutions for this function
    if len(resolved_args) == 0:
        diagnostics.summary(f"  No vectorization canidates found.")
        return None
    elif len(resolved_args) > 1:
        diagnostics.detail(f"  Multiple vectorization canidates found:")
        for ra in resolved_args:
            diagnostics.detail(f"  Candidate:")
            for i, arg in enumerate(ra):
                diagnostics.detail(
                    f"      {i}: {arg.python} -> {arg.vector.full_name if arg.vector else '<unresolved>'}"
                )

    elif len(resolved_args) == 1:
        diagnostics.detail(f"  Vectorization candidate found:")
        for i, arg in enumerate(resolved_args[0]):
            diagnostics.detail(
                f"    {i}: {arg.python} -> {arg.vector.full_name if arg.vector else '<unresolved>'}"
            )

    # If we got more than 1 resolution, try using slang's specialization system to narrow it down,
    # as slang may be able to use more precise generic rules to eliminate candiates that python
    # couldn't.
    # Skip this for fused functions since they don't have reflection
    if len(resolved_args) != 1:
        specialized_args = []
        for ra in resolved_args:
            slang_reflections = [cast("SlangType", arg.vector).type_reflection for arg in ra]
            specialized = function.reflection.specialize_with_arg_types(slang_reflections)
            if specialized:
                specialized_args.append(ra)
        if len(specialized_args) > 1:
            diagnostics.summary(
                f"  Ambiguous call - Slang could not identify a unique candidate for vectorization."
            )
            return None
        elif len(specialized_args) == 0:
            diagnostics.summary(
                f"  Slang compiler could not match the function signature to any vectorization candidate."
            )
            return None
        resolved_args = specialized_args

    # Should now have just 1 result
    resolved_args = resolved_args[0]

    res = ResolveResult()

    # Output positional python arguments
    pos_arg_types = []
    for i, arg in enumerate(resolved_args):
        pos_arg_types.append(cast("SlangType", arg.vector))
    res.args = tuple(pos_arg_types)

    # Output keyword python arguments
    for name, arg in bindings.kwargs.items():
        if name == "_result":
            continue
        res.kwargs[name] = cast("SlangType", resolved_args[arg.param_index].vector)

    # Build list of slang types, and then native type reflections, and specialize
    slang_types = tuple([cast("SlangType", arg.vector) for arg in resolved_args])
    slang_reflections = [arg.type_reflection for arg in slang_types]
    specialized = function.reflection.specialize_with_arg_types(slang_reflections)
    if specialized is None:
        diagnostics.summary(
            f"  Slang compiler could not match the function signature to the vectorization candidate:"
        )
        diagnostics.summary(f"    {function.name}({', '.join([t.full_name for t in slang_types])})")
        return None

    # Also output the slangfunction
    type_reflection = None if this_type is None else this_type.type_reflection
    res.function = bind_context.layout.find_function(specialized, type_reflection)
    res.params = [ResolvedParam(p, t) for p, t in zip(res.function.parameters, slang_types)]

    diagnostics.detail(f"  Selected slang function signature:")
    diagnostics.detail(f"    {function_reflection(res.function.reflection)}")

    return res


def _resolve_function_with_overloads(
    bind_context: "BindContext",
    function: "SlangFunction",
    bindings: "BoundCall",
    diagnostics: ResolutionDiagnostic,
    this_type: Optional["SlangType"] = None,
):
    # Should never get a fused function here.
    if function.is_overloaded:
        resolutions = [
            _resolve_function_internal(bind_context, f, bindings, diagnostics, this_type)
            for f in function.overloads
        ]
    else:
        resolutions = [
            _resolve_function_internal(bind_context, function, bindings, diagnostics, this_type)
        ]

    # Filter out any None resolutions
    resolutions = [r for r in resolutions if r is not None]
    if len(resolutions) != 1:
        diagnostics.summary(f"Unable to resolve function.")
        return None
    return resolutions[0]


def resolve_fuse_root_node(
    bind_context: "BindContext",
    function: "FusedFunction",
    bindings: "BoundCall",
    diagnostics: ResolutionDiagnostic,
    this_type: Optional["SlangType"] = None,
):
    # Map python arguments to slang parameters and get a set of ResolutionArgs back
    current_args = _assign_parameter_indices(
        bind_context, function, bindings, diagnostics, this_type  # type: ignore
    )
    if current_args is None:
        return None

    root_node = function._fuser.root_node

    # Can now assign bindings to root node inputs
    for i, arg in enumerate(current_args):
        root_node.inputs[i].binding = arg.binding

    # Perform recursive type resolution on the graph
    resolved_nodes: set[int] = set()
    max_iterations = len(root_node.children) * 2  # Prevent infinite loops
    iteration = 0

    def resolve_recursive(node: "FuseNode"):
        if id(node) in resolved_nodes:
            return True

        if node.function is not None:
            if _resolve_node(bind_context, node, diagnostics):
                resolved_nodes.add(id(node))
                return True
        elif node.subgraph is not None:
            raise NotImplementedError("Subgraph resolution not yet implemented")
        else:
            for child in node.children:
                for port in child.inputs:
                    if port.source is not None:
                        source_node, source_port_name = port.source
                        if source_node is node:
                            continue
                        if source_node is not None:
                            source_output = source_node.get_output(source_port_name)
                            if source_output.type is not None:
                                port.type = source_output.type
                        else:
                            source_input = node.get_input(source_port_name)
                            if source_input.type is not None:
                                port.type = source_input.type
                            elif source_input.binding is not None:
                                port.binding = source_input.binding

                resolve_recursive(child)

            for child in node.children:
                for port in child.inputs:
                    if port.source is not None:
                        source_node, source_port_name = port.source
                        if source_node is node:
                            continue
                        if source_node is not None:
                            pass
                        else:
                            source_input = node.get_input(source_port_name)
                            source_input.type = port.type

            for port in node.outputs:
                if port.type is None:
                    if port.source is not None:
                        source_node, source_port_name = port.source
                        if source_node is not None:
                            source_output = source_node.get_output(source_port_name)
                            if source_output.type is not None:
                                port.type = source_output.type
                        else:
                            source_input = node.get_input(source_port_name)
                            if source_input.type is not None:
                                port.type = source_input.type
                            elif source_input.binding is not None:
                                port.binding = source_input.binding

        return False

    resolve_recursive(root_node)

    res = ResolveResult()

    # Output positional python arguments
    pos_arg_types = []
    for i, arg in enumerate(root_node.inputs):
        pos_arg_types.append(cast("SlangType", arg.type))
    res.args = tuple(pos_arg_types)

    # Output keyword python arguments
    for name, arg in bindings.kwargs.items():
        if name == "_result":
            continue
        res.kwargs[name] = cast("SlangType", root_node.inputs[arg.param_index].type)

    res.function = function
    res.params = [
        ResolvedParam(p, arg.type) for p, arg in zip(function.parameters, root_node.inputs)
    ]

    for i, arg in enumerate(current_args):
        arg.binding.param_index = i

    return res


def _can_resolve_node(node: "FuseNode") -> bool:
    """
    Check if a node has all the necessary input bindings to be resolved.

    Args:
        node: The node to check

    Returns:
        True if the node can be resolved, False otherwise
    """
    for input_port in node.inputs:
        # Check if the input has a binding (either from parent or from another node)
        if input_port.binding is None and input_port.source is None:
            return False

        # If source is from another node, check if that node has resolved output types
        if input_port.source is not None:
            source_node, source_port_name = input_port.source
            if source_node is not None:
                source_output = source_node.get_output(source_port_name)
                if source_output.type is None:
                    return False

    return True


def _resolve_node(
    bind_context: "BindContext", node: "FuseNode", diagnostics: ResolutionDiagnostic
) -> bool:
    """
    Resolve a single node in the fused graph.

    This propagates bindings to the node's inputs, calls the function resolution,
    and propagates resolved types to the node's outputs.

    Args:
        bind_context: The binding context
        node: The node to resolve
        diagnostics: Diagnostic object for error reporting

    Returns:
        True if resolution succeeded, False otherwise
    """
    from slangpy.bindings import BoundCall, BoundVariable

    # If this node has a subgraph, we need to recurse into it
    if node.subgraph is not None:
        # TODO: Handle subgraph resolution
        diagnostics.detail(f"  Skipping subgraph node '{node.name}' (not yet implemented)")
        return False

    # If this node doesn't have a function, it's an error
    if node.function is None:
        diagnostics.summary(f"  Node '{node.name}' has no function or subgraph")
        return False

    # Build BoundCall from the node's input ports
    # Create BoundVariable objects for each input
    bound_args: list[BoundVariable] = []

    for input_port in node.inputs:
        # Get the binding for this input
        if input_port.binding is not None:
            # Direct binding from parent node
            bound_var = input_port.binding
        elif input_port.source is not None:
            # Binding from another child node's output
            source_node, source_port_name = input_port.source
            if source_node is None:
                diagnostics.summary(f"  Invalid source for input '{input_port.name}'")
                return False

            source_output = source_node.get_output(source_port_name)
            if source_output.type is None:
                diagnostics.summary(f"  Source output '{source_port_name}' has no resolved type")
                return False

            # Create a BoundVariable with the resolved type
            # We need to create a synthetic binding here
            bound_var = BoundVariable(bind_context, None, None, input_port.name)
            bound_var.vector_type = source_output.type
            bound_var.slang_type = source_output.type
        else:
            diagnostics.summary(f"  Input '{input_port.name}' has no binding or source")
            return False

        bound_args.append(bound_var)

    # Create BoundCall with the bound variables
    bindings = BoundCall(bind_context)
    bindings.args = bound_args
    bindings.kwargs = {}

    # Get the SlangFunction from the node's function
    slang_func = node.function._slang_func

    # Check if this is a fused function - if so, we need to handle it differently
    from slangpy.experimental.fuse import FusedFunction

    if isinstance(slang_func, FusedFunction):
        # Recursive fused function - need to resolve it
        diagnostics.detail(
            f"  Node '{node.name}' contains a nested fused function (not yet supported)"
        )
        return False

    # Resolve the function with the bindings
    diagnostics.detail(f"  Resolving node '{node.name}' with function '{slang_func.name}'")

    result = _resolve_function_with_overloads(bind_context, slang_func, bindings, diagnostics, None)

    if result is None:
        diagnostics.summary(f"  Failed to resolve node '{node.name}'")
        return False

    # Propagate resolved types to input ports
    for i, input_port in enumerate(node.inputs):
        if i < len(result.args):
            input_port.type = result.args[i]

    # Propagate return type to output port (assuming single output)
    if len(node.outputs) > 0 and result.function.return_type is not None:
        node.outputs[0].type = result.function.return_type

    diagnostics.detail(
        f"  Successfully resolved node '{node.name}' -> output type: "
        f"{node.outputs[0].type.full_name if node.outputs and node.outputs[0].type else 'void'}"
    )

    return True


def resolve_function(
    bind_context: "BindContext",
    function: "SlangFunction",
    bindings: "BoundCall",
    diagnostics: ResolutionDiagnostic,
    this_type: Optional["SlangType"] = None,
):
    from slangpy.experimental.fuse import FusedFunction

    if isinstance(function, FusedFunction):
        assert this_type is None

        # Got a fused function. Need to perform recursive operation in which
        # leaf nodes are specialized first, then the fused function itself is specialized
        specialized_function = copy.deepcopy(function)
        specialized_function._fuser.clear_type_info()
        specialized_function._fuser.sort_graph()

        print("Before resolving fused function:")
        print(specialized_function._fuser.dump_graph())

        res = resolve_fuse_root_node(
            bind_context, specialized_function, bindings, diagnostics, this_type
        )

        print("After resolving fused function:")
        print(specialized_function._fuser.dump_graph())

        return res
    else:
        # Not a fused function, proceed as normal
        return _resolve_function_with_overloads(
            bind_context, function, bindings, diagnostics, this_type
        )
