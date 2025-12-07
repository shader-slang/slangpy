# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Type Resolution System

This module resolves Python arguments to Slang function parameters, handling:
- Mapping positional/keyword arguments to Slang parameters
- Resolving Python types to Slang types via marshals
- Disambiguating overloads and generic specializations
- Handling special cases ('this', '_result')

Architecture:
    1. ArgumentSpec/ArgumentMapping: Lightweight input structures
    2. map_arguments_to_parameters(): Maps Python args to Slang params
    3. resolve_argument_list(): Resolves types using marshals
    4. disambiguate_with_slang(): Narrows candidates via Slang specialization
    5. resolve_function_call(): High-level orchestration pipeline
    6. _resolve_function_internal(): Backward-compatible adapter for BoundCall

Public API:
    - resolve_function(): Main entry point (handles overloads)
    - resolve_function_call(): New clean API (no BoundCall dependency)
    - ResolveResult: Output containing resolved types and specialized function
"""
from typing import TYPE_CHECKING, Any, Optional, Union, cast
from slangpy.core.native import CallMode
from slangpy.core.logging import function_reflection

if TYPE_CHECKING:
    from slangpy.reflection.reflectiontypes import SlangType, SlangFunction, SlangParameter
    from slangpy.bindings import BindContext, BoundCall, BoundVariable
    from slangpy.core.native import NativeMarshall
    from slangpy.experimental.fuse import FuseNode, FusedFunction


# ============================================================================
# Data Structures
# ============================================================================


class ArgumentSpec:
    """
    Input specification for a single argument to resolve.
    Contains the Python marshal and Slang parameter type.
    """

    def __init__(
        self,
        python_marshal: Optional["NativeMarshall"],
        slang_param: "SlangParameter",
        known_type: Optional["SlangType"] = None,
    ):
        super().__init__()
        self.python_marshal = python_marshal
        self.slang_param = slang_param
        self.known_type = known_type


class ResolvedArgument:
    """Output of resolving a single argument."""

    def __init__(self, spec: ArgumentSpec, resolved_type: "SlangType"):
        super().__init__()
        self.spec = spec
        self.resolved_type = resolved_type


class ArgumentMapping:
    """Result of mapping python arguments to slang parameters."""

    def __init__(self):
        super().__init__()
        self.param_specs: list[ArgumentSpec] = []
        self.this_marshal: Union[Optional["NativeMarshall"], "SlangType"] = None
        self.result_marshal: Union[Optional["NativeMarshall"], "SlangType"] = None
        self.errors: list[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


# ============================================================================
# Legacy Data Structures (for backward compatibility)
# ============================================================================


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


# ============================================================================
# Argument Mapping
# ============================================================================


def map_arguments_to_parameters(
    function: "SlangFunction",
    args: list[Union[Optional["NativeMarshall"], "SlangType"]],
    kwargs: dict[str, Union[Optional["NativeMarshall"], "SlangType"]],
    call_mode: CallMode,
    this_type: Optional["SlangType"] = None,
) -> ArgumentMapping:
    """
    Maps Python positional/keyword arguments to Slang function parameters.
    Handles special cases: 'this' (first arg for non-static methods) and '_result' (last arg for derivatives).

    Arguments can be either:
    - NativeMarshall (or None): Needs type resolution
    - SlangType: Already resolved, used directly as known_type
    """
    mapping = ArgumentMapping()
    function_parameters = list(function.parameters)

    # Determine special argument positions
    first_arg_is_this = (
        this_type is not None and not function.static and not function.is_constructor
    )
    last_arg_is_retval = (
        function.return_type is not None
        and function.return_type.name != "void"
        and "_result" not in kwargs
        and call_mode != CallMode.prim
    )

    # Extract special arguments
    regular_args = list(args)
    if first_arg_is_this:
        if len(regular_args) == 0:
            mapping.errors.append("Expected 'this' argument for non-static member function")
            return mapping
        mapping.this_marshal = regular_args[0]
        regular_args = regular_args[1:]

    if last_arg_is_retval:
        if len(regular_args) == 0:
            mapping.errors.append("Expected '_result' argument for derivative call")
            return mapping
        mapping.result_marshal = regular_args[-1]
        regular_args = regular_args[:-1]

    # Check argument count
    if len(regular_args) > len(function_parameters):
        mapping.errors.append(
            f"Too many arguments: expected {len(function_parameters)}, got {len(regular_args)}"
        )
        return mapping

    # Build positional parameter list
    positioned_args: list[Union[Optional["NativeMarshall"], "SlangType"]] = [None] * len(
        function_parameters
    )
    for i, arg in enumerate(regular_args):
        positioned_args[i] = arg

    # Apply keyword arguments
    name_map = {param.name: i for i, param in enumerate(function_parameters)}
    for name, arg in kwargs.items():
        if name == "_result":
            continue
        if name not in name_map:
            mapping.errors.append(f"No parameter named '{name}'")
            return mapping
        i = name_map[name]
        if positioned_args[i] is not None:
            mapping.errors.append(f"Parameter '{name}' specified multiple times")
            return mapping
        positioned_args[i] = arg

    # Verify all parameters are specified
    for i, arg in enumerate(positioned_args):
        if arg is None:
            mapping.errors.append(f"Parameter '{function_parameters[i].name}' not specified")
            return mapping

    # Build ArgumentSpec list
    # If arg is already a SlangType, use it as known_type; otherwise it's a marshal
    from slangpy.reflection.reflectiontypes import SlangType

    for arg, param in zip(positioned_args, function_parameters):
        if isinstance(arg, SlangType):
            mapping.param_specs.append(ArgumentSpec(None, param, known_type=arg))
        else:
            mapping.param_specs.append(ArgumentSpec(arg, param))

    return mapping


# ============================================================================
# Type Resolution Core
# ============================================================================


def resolve_single_argument(
    bind_context: "BindContext", spec: ArgumentSpec, diagnostics: ResolutionDiagnostic
) -> list[Any]:
    """
    Resolve a single argument to all possible Slang types based on its marshal and parameter type.
    Returns empty list if no valid types found.
    """
    if spec.known_type:
        return [spec.known_type]

    if not spec.python_marshal:
        return [spec.slang_param.type]

    python = spec.python_marshal
    slang = spec.slang_param.type

    if hasattr(python, "resolve_types"):
        types = python.resolve_types(bind_context, slang)
    else:
        types = [python.resolve_type(bind_context, slang)]

    if not types:
        types = []
    types = [t for t in types if t]

    if len(types) == 0:
        diagnostics.summary(
            f"  Argument '{spec.slang_param.name}' could not be resolved: "
            f"python type {python} does not match slang type {slang.full_name}"
        )

    return types


def resolve_argument_list(
    bind_context: "BindContext", specs: list[ArgumentSpec], diagnostics: ResolutionDiagnostic
) -> list[list[ResolvedArgument]]:
    """
    Resolve multiple arguments to all possible type combinations.
    Returns list of possible resolutions (each is a list of ResolvedArgument).
    """
    resolutions: list[list[ResolvedArgument]] = [[]]

    for spec in specs:
        possible_types = resolve_single_argument(bind_context, spec, diagnostics)

        new_resolutions: list[list[ResolvedArgument]] = []
        for resolved_type in possible_types:
            for resolution in resolutions:
                new_resolution = resolution + [ResolvedArgument(spec, resolved_type)]
                new_resolutions.append(new_resolution)
        resolutions = new_resolutions

    return resolutions


# ============================================================================
# Disambiguation
# ============================================================================


def disambiguate_with_slang(
    function: "SlangFunction", resolutions: list[list[ResolvedArgument]]
) -> list[list[ResolvedArgument]]:
    """
    Use Slang's specialization system to eliminate invalid resolution candidates.
    Returns only those resolutions that Slang can successfully specialize.
    """
    valid_resolutions: list[list[ResolvedArgument]] = []

    for resolution in resolutions:
        slang_types = [arg.resolved_type.type_reflection for arg in resolution]
        specialized = function.reflection.specialize_with_arg_types(slang_types)
        if specialized:
            valid_resolutions.append(resolution)

    return valid_resolutions


def select_unique_resolution(
    resolutions: list[list[ResolvedArgument]], diagnostics: ResolutionDiagnostic
) -> Optional[list[ResolvedArgument]]:
    """
    Ensure exactly one resolution remains, logging diagnostics if not.
    """
    if len(resolutions) == 0:
        diagnostics.summary("  No valid type resolution found")
        return None

    if len(resolutions) > 1:
        diagnostics.summary("  Ambiguous call - multiple valid resolutions found")
        diagnostics.detail(f"  Found {len(resolutions)} possible resolutions:")
        for i, resolution in enumerate(resolutions):
            types_str = ", ".join([arg.resolved_type.full_name for arg in resolution])
            diagnostics.detail(f"    Resolution {i + 1}: {types_str}")
        return None

    return resolutions[0]


# ============================================================================
# Result Building
# ============================================================================


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


def build_resolve_result(
    bind_context: "BindContext",
    function: "SlangFunction",
    resolved_args: list[ResolvedArgument],
    kwargs_dict: dict[str, int],
    this_type: Optional["SlangType"],
    diagnostics: ResolutionDiagnostic,
) -> Optional[ResolveResult]:
    """
    Build the final ResolveResult from resolved arguments.
    Specializes the function with the resolved types.
    """
    result = ResolveResult()

    # Extract resolved types
    slang_types = tuple([arg.resolved_type for arg in resolved_args])
    result.slang = slang_types

    # Build positional args (excluding kwargs)
    result.args = tuple([resolved_args[i].resolved_type for i in range(len(resolved_args))])

    # Build kwargs mapping
    for name, idx in kwargs_dict.items():
        result.kwargs[name] = resolved_args[idx].resolved_type

    # Specialize the function with resolved types
    slang_reflections = [t.type_reflection for t in slang_types]
    specialized = function.reflection.specialize_with_arg_types(slang_reflections)
    if specialized is None:
        diagnostics.summary("  Slang compiler could not specialize function with resolved types:")
        diagnostics.summary(f"    {function.name}({', '.join([t.full_name for t in slang_types])})")
        return None

    # Find the specialized function
    type_reflection = None if this_type is None else this_type.type_reflection
    result.function = bind_context.layout.find_function(specialized, type_reflection)
    result.params = [ResolvedParam(p, t) for p, t in zip(result.function.parameters, slang_types)]

    diagnostics.detail(f"  Selected slang function signature:")
    diagnostics.detail(f"    {function_reflection(result.function.reflection)}")

    return result


# ============================================================================
# High-Level Orchestration
# ============================================================================


def resolve_function_call(
    bind_context: "BindContext",
    function: "SlangFunction",
    args: list[Union[Optional["NativeMarshall"], "SlangType"]],
    kwargs: dict[str, Union[Optional["NativeMarshall"], "SlangType"]],
    diagnostics: ResolutionDiagnostic,
    this_type: Optional["SlangType"] = None,
) -> Optional[ResolveResult]:
    """
    Main resolution pipeline:
    1. Map arguments to parameters
    2. Resolve types for each parameter
    3. Disambiguate using Slang specialization
    4. Build final result

    Arguments can be either:
    - NativeMarshall (or None): Needs type resolution
    - SlangType: Already resolved, used directly as known_type
    """
    diagnostics.summary(f"{function_reflection(function.reflection)}")

    # Log argument info
    if len(args) > 0:
        diagnostics.detail(f"  Positional python arguments:")
        for i, arg in enumerate(args):
            if arg is not None:
                diagnostics.detail(f"    {i}: {arg}")
    if len(kwargs) > 0:
        diagnostics.detail(f"  Named python arguments:")
        for name, arg in kwargs.items():
            if arg is not None:
                diagnostics.detail(f"    {name}: {arg}")

    # Step 1: Map arguments to parameters
    mapping = map_arguments_to_parameters(function, args, kwargs, bind_context.call_mode, this_type)
    if not mapping.is_valid:
        for error in mapping.errors:
            diagnostics.summary(f"  {error}")
        return None

    # Step 2: Resolve types for each parameter
    resolutions = resolve_argument_list(bind_context, mapping.param_specs, diagnostics)
    if len(resolutions) == 0:
        diagnostics.summary(f"  No vectorization candidates found")
        return None

    # Log resolution candidates
    if len(resolutions) > 1:
        diagnostics.detail(f"  Multiple vectorization candidates found:")
        for resolution in resolutions:
            types_str = ", ".join([arg.resolved_type.full_name for arg in resolution])
            diagnostics.detail(f"    Candidate: {types_str}")
    elif len(resolutions) == 1:
        diagnostics.detail(f"  Vectorization candidate found:")
        types_str = ", ".join([arg.resolved_type.full_name for arg in resolutions[0]])
        diagnostics.detail(f"    {types_str}")

    # Step 3: Disambiguate if needed
    if len(resolutions) > 1:
        resolutions = disambiguate_with_slang(function, resolutions)
        if len(resolutions) == 0:
            diagnostics.summary(
                "  Slang compiler could not match function signature to any vectorization candidate"
            )
            return None

    resolved = select_unique_resolution(resolutions, diagnostics)
    if resolved is None:
        return None

    # Step 4: Build result
    # Build kwargs index mapping for result construction
    kwargs_indices = {}
    param_names = [p.name for p in function.parameters]
    for name in kwargs:
        if name != "_result":
            kwargs_indices[name] = param_names.index(name)

    return build_resolve_result(
        bind_context, function, resolved, kwargs_indices, this_type, diagnostics
    )


# ============================================================================
# Backward Compatibility Adapter
# ============================================================================


def _resolve_function_internal(
    bind_context: "BindContext",
    function: "SlangFunction",
    bindings: "BoundCall",
    diagnostics: ResolutionDiagnostic,
    this_type: Optional["SlangType"] = None,
) -> Optional[ResolveResult]:
    """
    Backward-compatible adapter that extracts marshals from BoundCall,
    calls resolve_function_call, and updates param_index fields in BoundVariables.
    """
    assert not function.is_overloaded

    # Extract marshals or vector_types from BoundVariables
    # If vector_type is set (from explicit .map()), use it directly as known_type
    args: list[Union[Optional["NativeMarshall"], "SlangType"]] = [
        v.vector_type if v.vector_type is not None else v.python for v in bindings.args
    ]
    kwargs: dict[str, Union[Optional["NativeMarshall"], "SlangType"]] = {
        k: v.vector_type if v.vector_type is not None else v.python
        for k, v in bindings.kwargs.items()
    }

    # Call new implementation
    result = resolve_function_call(bind_context, function, args, kwargs, diagnostics, this_type)

    if result is None:
        return None

    # Update param_index fields in BoundVariables
    # Determine special argument positions
    first_arg_is_this = (
        this_type is not None and not function.static and not function.is_constructor
    )
    last_arg_is_retval = (
        function.return_type is not None
        and function.return_type.name != "void"
        and "_result" not in bindings.kwargs
        and bind_context.call_mode != CallMode.prim
    )

    # Extract regular args (excluding 'this' and '_result')
    regular_args = list(bindings.args)
    if first_arg_is_this:
        regular_args[0].param_index = -1
        regular_args = regular_args[1:]
    if last_arg_is_retval:
        regular_args[-1].param_index = len(function.parameters)
        regular_args = regular_args[:-1]

    # Assign param_index for positional arguments
    for i, arg in enumerate(regular_args):
        arg.param_index = i

    # Assign param_index for keyword arguments
    name_map = {param.name: i for i, param in enumerate(function.parameters)}
    for name, arg in bindings.kwargs.items():
        if name == "_result":
            continue
        arg.param_index = name_map[name]

    return result


def resolve_function(
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
