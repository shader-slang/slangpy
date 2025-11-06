# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Optional, cast
from slangpy.bindings import BindContext, BoundCall, BoundVariable
from slangpy.reflection.reflectiontypes import SlangType, SlangFunction, SlangParameter
from slangpy.core.native import CallMode, NativeMarshall
from slangpy import FunctionReflection


class ResolutionArg:
    """
    Holds a single argument for type resolution. Starts with the python marshall and slang type
    being resolved, and ends with the resolved vector type (if any). In the case of functions
    with optional parameters, the python variable can be None and the parameter type will be used
    """

    def __init__(self):
        super().__init__()
        self.slang: SlangType = None  # type: ignore
        self.vector: Optional[SlangType] = None
        self.python: Optional[NativeMarshall] = None


def create_arg(variable: Optional[BoundVariable], param: SlangParameter):
    arg = ResolutionArg()
    arg.slang = param.type
    if variable:
        arg.python = variable.python
        arg.vector = None
    else:
        arg.python = None
        arg.vector = arg.slang
    return arg


def clone_args(args: list[ResolutionArg]):
    new_args = []
    for arg in args:
        new_arg = ResolutionArg()
        new_arg.python = arg.python
        new_arg.slang = arg.slang
        new_arg.vector = arg.vector
        new_args.append(new_arg)
    return new_args


def resolve_arguments(bind_context: BindContext, args: list[ResolutionArg]):
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

        new_resolutions = []
        for potential_type in types:
            if not potential_type:
                continue
            for resolution in resolutions:
                new_args = clone_args(resolution)
                new_args[i].vector = cast(SlangType, potential_type)
                new_resolutions.append(new_args)
        resolutions = new_resolutions

    return resolutions


class ResolveResult:
    def __init__(self):
        super().__init__()
        self.args: tuple[SlangType, ...] = ()
        self.kwargs: dict[str, SlangType] = {}
        self.slang: tuple[SlangType, ...] = ()
        self.reflection: FunctionReflection = None  # type: ignore


def resolve_function(
    bind_context: BindContext,
    function: SlangFunction,
    bindings: BoundCall,
    this_type: Optional[SlangType] = None,
):

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
    positioned_args: list[Optional[BoundVariable]] = [None] * len(function_parameters)

    # Populate the first N arguments from provided positional arguments
    if len(signature_args) > len(function_parameters):
        return False
    for i, arg in enumerate(signature_args):
        positioned_args[i] = arg
        arg.param_index = i

    # Attempt to populate the remaining arguments from keyword arguments
    name_map = {param.name: i for i, param in enumerate(function_parameters)}
    for name, arg in signature_kwargs.items():
        if name == "_result":
            continue
        if name not in name_map:
            return False
        i = name_map[name]
        if positioned_args[i] is not None:
            return False
        positioned_args[i] = arg
        arg.param_index = i

    # If we reach this point, all positional and keyword arguments have been matched to slang parameters
    # Now create a set of tuples that are [MarshallType, ParameterType, ResolvedType|None]
    current_args = [
        create_arg(param, arg) for param, arg in zip(positioned_args, function_parameters)
    ]

    # Use resolve_arguments to try and find 1 or more possible resolutions
    resolved_args = resolve_arguments(bind_context, current_args)

    # No valid resolutions for this function
    if len(resolved_args) == 0:
        return None

    # If we got more than 1 resolution, try using slang's specialization system to narrow it down,
    # as slang may be able to use more precise generic rules to eliminate candiates that python
    # couldn't.
    if len(resolved_args) != 1:
        specialized_args = []
        for ra in resolved_args:
            slang_reflections = [cast(SlangType, arg.vector).type_reflection for arg in ra]
            specialized = function.reflection.specialize_with_arg_types(slang_reflections)
            if specialized:
                specialized_args.append(ra)
        if len(specialized_args) != 1:
            return None
        resolved_args = specialized_args

    # Should now have just 1 result
    resolved_args = resolved_args[0]

    res = ResolveResult()

    # Output positional python arguments
    pos_arg_types = []
    for i, arg in enumerate(resolved_args):
        pos_arg_types.append(cast(SlangType, arg.vector))
    res.args = tuple(pos_arg_types)

    # Output keyword python arguments
    for name, arg in signature_kwargs.items():
        if name == "_result":
            continue
        i = name_map[name]
        res.kwargs[name] = cast(SlangType, resolved_args[i].vector)

    # Output positional slang arguments
    res.slang = tuple([cast(SlangType, arg.vector) for arg in resolved_args])

    # Re-specialize the function to let slang do a final check and get the final
    # concrete reflection with a concrete return value.
    slang_reflections = [arg.type_reflection for arg in res.slang]
    specialized = function.reflection.specialize_with_arg_types(slang_reflections)
    res.reflection = specialized

    return res
