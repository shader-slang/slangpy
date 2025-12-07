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
from typing import TYPE_CHECKING, Any, Optional, cast
from slangpy.core.native import CallMode
from slangpy.core.logging import function_reflection

if TYPE_CHECKING:
    from slangpy.reflection.reflectiontypes import SlangType, SlangFunction, SlangParameter
    from slangpy.bindings import BindContext, BoundCall, BoundVariable
    from slangpy.core.native import NativeMarshall


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
        self.this_marshal: Optional["NativeMarshall"] = None
        self.result_marshal: Optional["NativeMarshall"] = None
        self.errors: list[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


# ============================================================================
# Legacy Data Structures (for backward compatibility)
# ============================================================================


class ResolutionArg:
    """
    Holds a single argument for type resolution. Starts with the python marshall and slang type
    being resolved, and ends with the resolved vector type (if any). In the case of functions
    with optional parameters, the python variable can be None and the parameter type will be used
    """

    def __init__(self):
        super().__init__()
        self.slang: "SlangType" = None  # type: ignore
        self.resolved: Optional["SlangType"] = None
        self.python: Optional["NativeMarshall"] = None


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
    args: list[Optional["NativeMarshall"]],
    kwargs: dict[str, Optional["NativeMarshall"]],
    call_mode: CallMode,
    this_type: Optional["SlangType"] = None,
) -> ArgumentMapping:
    """
    Maps Python positional/keyword arguments to Slang function parameters.
    Handles special cases: 'this' (first arg for non-static methods) and '_result' (last arg for derivatives).
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
    positioned_marshals: list[Optional["NativeMarshall"]] = [None] * len(function_parameters)
    for i, arg in enumerate(regular_args):
        positioned_marshals[i] = arg

    # Apply keyword arguments
    name_map = {param.name: i for i, param in enumerate(function_parameters)}
    for name, arg in kwargs.items():
        if name == "_result":
            continue
        if name not in name_map:
            mapping.errors.append(f"No parameter named '{name}'")
            return mapping
        i = name_map[name]
        if positioned_marshals[i] is not None:
            mapping.errors.append(f"Parameter '{name}' specified multiple times")
            return mapping
        positioned_marshals[i] = arg

    # Verify all parameters are specified
    for i, marshal in enumerate(positioned_marshals):
        if marshal is None:
            mapping.errors.append(f"Parameter '{function_parameters[i].name}' not specified")
            return mapping

    # Build ArgumentSpec list
    for marshal, param in zip(positioned_marshals, function_parameters):
        mapping.param_specs.append(ArgumentSpec(marshal, param))

    return mapping


# ============================================================================
# Legacy Helper Functions (for old resolve_arguments implementation)
# ============================================================================
# These functions support the legacy resolve_arguments() which operates on
# ResolutionArg. They are kept for reference but not used by the new pipeline.


def create_arg(variable: Optional["BoundVariable"], param: "SlangParameter"):
    arg = ResolutionArg()
    arg.slang = param.type
    if variable:
        arg.python = variable.python
        arg.resolved = variable.vector_type
    else:
        arg.python = None
        arg.resolved = arg.slang
    return arg


def clone_args(args: list[ResolutionArg]):
    new_args = []
    for arg in args:
        new_arg = ResolutionArg()
        new_arg.python = arg.python
        new_arg.slang = arg.slang
        new_arg.resolved = arg.resolved
        new_args.append(new_arg)
    return new_args


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
# Legacy Resolution Function
# ============================================================================


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
        python, slang, vec = args[i].python, args[i].slang, args[i].resolved
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
                new_args[i].resolved = cast("SlangType", potential_type)
                new_resolutions.append(new_args)
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
    args: list[Optional["NativeMarshall"]],
    kwargs: dict[str, Optional["NativeMarshall"]],
    diagnostics: ResolutionDiagnostic,
    this_type: Optional["SlangType"] = None,
) -> Optional[ResolveResult]:
    """
    Main resolution pipeline:
    1. Map arguments to parameters
    2. Resolve types for each parameter
    3. Disambiguate using Slang specialization
    4. Build final result
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

    # Extract marshals from BoundVariables
    args: list[Optional["NativeMarshall"]] = [v.python for v in bindings.args]
    kwargs: dict[str, Optional["NativeMarshall"]] = {
        k: v.python for k, v in bindings.kwargs.items()
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
