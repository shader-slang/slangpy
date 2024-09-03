import hashlib
from io import StringIO
from typing import Any, Optional, Union, cast
from sgl import FunctionReflection
from kernelfunctions.shapes import TConcreteShape
from kernelfunctions.signaturenode import CallMode, SignatureNode, TCallSignature, TMatchedSignature
from kernelfunctions.typeregistry import PYTHON_SIGNATURE_HASH, AccessType, create_slang_type_marshal


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


def build_signature(*args: Any, **kwargs: Any) -> TCallSignature:
    """
    Builds a basic call signature for a given set of python 
    arguments and keyword arguments
    """
    arg_signatures = [SignatureNode(x) for x in args]
    kwarg_signatures = {k: SignatureNode(v) for k, v in kwargs.items()}
    return (arg_signatures, kwarg_signatures)


def match_signature(
    signature: TCallSignature,
    function_reflection: FunctionReflection,
    call_mode: CallMode
) -> Union[None, TMatchedSignature]:
    """
    Attempts to efficiently match a signature to a slang function overload.
    Returns a dictionary of matched argument nodes to parameter names.
    """

    overload_parameters = [x for x in function_reflection.parameters]

    args = signature[0]
    kwargs = signature[1]
    rval = None
    matched_rval: Optional[SignatureNode] = None

    # Check for user providing return value. In all cases, it
    # can be explicitly passed as a keyword argument. With derivative
    # calls, if not provided as keyword, it is EXPECTED to be
    # the last positional argument.
    if "_result" in kwargs:
        rval = kwargs["_result"]
        del kwargs["_result"]
        if not rval.is_compatible(function_reflection):
            return None
    elif call_mode != CallMode.prim and function_reflection.return_type is not None:
        rval = args[-1]
        args = args[:-1]
        if not rval.is_compatible(function_reflection):
            return None

    # If there are more positional arguments than parameters, it can't match.
    if len(args) > len(overload_parameters):
        return None

    # Dictionary of slang arguments and corresponding python arguments
    param_name_to_index = {x.name: i for i, x in enumerate(overload_parameters)}
    matched_params: dict[str, Optional[SignatureNode]] = {
        x.name: None for x in overload_parameters}

    # Positional arguments must all match perfectly
    for i, arg in enumerate(args):
        param = overload_parameters[i]
        if not arg.is_compatible(param):
            return None
        arg.param_index = i
        matched_params[param.name] = arg

    # Pair up kw arguments with slang arguments
    for name, arg in kwargs.items():
        i = param_name_to_index.get(name)
        if i is None:
            return None
        param = overload_parameters[i]
        if not arg.is_compatible(param):
            return None
        arg.param_index = i
        matched_params[param.name] = arg

    # Check if all arguments have been handled
    for param in matched_params.values():
        if arg is None:
            return None

    # Need to add something to handle return value here
    if call_mode == CallMode.prim and rval is None:
        matched_rval = SignatureNode(None)
        matched_rval.slang_type = create_slang_type_marshal(
            function_reflection.return_type)

    if rval is not None:
        matched_params["_result"] = rval
    return matched_params  # type: ignore


def apply_signature(
        signature: TMatchedSignature,
        function_reflection: FunctionReflection,
        input_transforms: Optional[dict[str, TConcreteShape]] = None,
        output_transforms: Optional[dict[str, TConcreteShape]] = None):
    """
    Apply a matched signature to a slang function, adding slang type marshalls
    to the signature nodes and performing other work that kicks in once
    match has occured.
    """
    for name, node in signature.items():
        reflection = function_reflection if name == "_result" else function_reflection.parameters[
            node.param_index]
        node.apply_signature(reflection, name, input_transforms, output_transforms)


def calculate_and_apply_call_shape(signature: TMatchedSignature) -> list[int]:
    """
    Given the shapes of the parameters (inferred from reflection) and inputs (passed in by the user), 
    calculates the argument shapes and call shape.
    All parameters must have a shape, however individual dimensions can have an undefined size (None).
    Inputs can also be fully undefined.
    """

    # Get all arguments that are to be written
    nodes: list[SignatureNode] = []
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

    return verified_call_shape


def generate_call_data_struct(call_shape: TConcreteShape, signature: TMatchedSignature, mode: CallMode):
    inputs: list[Any] = []
    generate_call_data_declarations(call_shape, signature, mode, inputs)
    combined = "struct CallData {\n    " + \
        (";\n    ").join(inputs) + ";\n}\nParameterBlock<CallData> call_data;\n"
    return combined


def generate_call_data_declarations(call_shape: TConcreteShape, signature: TMatchedSignature, mode: CallMode, out_inputs: list[Any]):
    """
    Generate a list of call data nodes that will be used to generate the call
    """
    nodes: list[SignatureNode] = []
    for node in signature.values():
        node.get_input_list(nodes)

    for node in nodes:
        arg_name = node.path.replace(".", "__")
        prim_type_name = node.slang_type.name
        diff_type_name = None if node.slang_differential is None else node.slang_differential.name

        if mode == CallMode.prim:
            node.python_marshal.declare_inputs(node.path.replace(".", "__"),
                                               # type: ignore (already checked)
                                               node.argument_shape,
                                               prim_type_name,
                                               node.prim_access,
                                               None,
                                               AccessType.none,
                                               out_inputs)
        elif mode == CallMode.bwds:
            node.python_marshal.declare_inputs(node.path.replace(".", "__"),
                                               # type: ignore (already checked)
                                               node.argument_shape,
                                               prim_type_name,
                                               node.bwds_access[0],
                                               diff_type_name,
                                               node.bwds_access[1],
                                               out_inputs)
        else:
            node.python_marshal.declare_inputs(node.path.replace(".", "__"),
                                               # type: ignore (already checked)
                                               node.argument_shape,
                                               prim_type_name,
                                               node.fwds_access[0],
                                               diff_type_name,
                                               node.fwds_access[1],
                                               out_inputs)


'''
So lets assume we have general code that at any given level basically has a set of named
variables. arguably that should be the case for the root as well. So in a simple case like

add(a,b)

we'd expect a node tree of 
{
    a: SignatureNode,
    b: SignatureNode
}

you could argue that root should be a signature node really

SignatureNode({
    a: SignatureNode,
    b: SignatureNode
})

So that node represents a set of values etc - it is the trampoline call and the arguments
that feed into it?

What if it was

SignatureNode({
    a: SignatureNode({
        x: SignatureNode,
        y: SignatureNode
    }),
    b: SignatureNode,
})

now to load a we have to call a function, and to store a we have to call a function.

so for the simple example it was kind of

main:
    a = load_a()
    b = load_b()
    call(a,b)
    store_a(a)
    store_b(b)

load_a:
    return call_data.a

store_a
    call_data.a = a

for the more complex example we need
main:
    a = load_a()
    b = load_b()
    call(a,b)
    store_a(a)
    store_b(b)

load_a:
    A a
    a.x = load_x()
    a.y = load_y()
    return a

store_a:
    store_x(a.x)
    store_y(a.y)

So the only difference really is that the root node contains loads and stores into
local variables and has a call, where the child nodes are building structures    

SO I might need a slang FIELD marshall really, as it needs to know its type and name,
because it'll want to know where to load from (maybe?). Then again maybe not.

Arguably not, though it'd probably be useful to do so anyway to make it a bit
more human readable

In the context of types, you end up with just load_self, and store_self, and then
it all just works

'''
