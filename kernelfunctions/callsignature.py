from enum import Enum
import hashlib
from io import StringIO
from typing import Any, Callable, Optional, Union, cast
from sgl import FunctionReflection, TypeReflection, VariableReflection

from kernelfunctions.shapes import TConcreteOrUndefinedShape, TConcreteShape, TLooseOrUndefinedShape, TLooseShape


class CallMode(Enum):
    prim = 0
    bwds = 1
    fwds = 2


# Result of building the signature for a set of args and kwargs
# passed as part of a python call
TCallSignature = tuple[list['SignatureNode'], dict[str, 'SignatureNode']]

# Result of matching a signature to a slang function, tuple
# with set of positional arguments and optional return value
TMatchedSignature = dict[str, 'SignatureNode']

# Base class for marshalling slang types


class BaseSlangTypeMarshal:
    def __init__(self, slang_type: TypeReflection):
        super().__init__()
        self.name = slang_type.full_name
        self.value_shape: TLooseShape = ()
        self.container_shape: TLooseShape = ()

    @property
    def shape(self):
        return self.container_shape + self.value_shape

    def __repr__(self):
        return self.name


class ScalarSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (1,)


class VectorSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (slang_type.col_count,)


class MatrixSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (slang_type.row_count, slang_type.col_count)


class StructSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (1,)


# Base class for marshalling python types
class BasePythonTypeMarshal:
    def __init__(self, python_type: type):
        super().__init__()
        self.type = python_type
        self.shape: TLooseOrUndefinedShape = None

    def is_compatible(self, slang_type: TypeReflection) -> bool:
        raise NotImplementedError()

    @property
    def name(self):
        return self.type.__name__

    def __repr__(self):
        return self.type.__name__


# Dictionary of python types to corresponding marshall
PYTHON_TYPE_MARSHAL: dict[type, Callable[[Any], BasePythonTypeMarshal]] = {}

# Dictionary of python types to corresponding hash functions
PYTHON_SIGNATURE_HASH: dict[type, Optional[Callable[[StringIO, Any], Any]]] = {
    int: None,
    float: None,
    bool: None,
    str: None,
    list: None,
    dict: None,
    tuple: None,
}

# Register a mapping from type to marshall


def register_python_type(
    python_type: type,
    marshall: Union[Callable[[Any], BasePythonTypeMarshal], BasePythonTypeMarshal],
    hash_fn: Optional[Callable[[StringIO, Any], Any]],
):
    if isinstance(marshall, BasePythonTypeMarshal):
        def cb(x: Any) -> BasePythonTypeMarshal:
            return cast(BasePythonTypeMarshal, marshall)
        PYTHON_TYPE_MARSHAL[python_type] = cb
    else:
        PYTHON_TYPE_MARSHAL[python_type] = marshall
    PYTHON_SIGNATURE_HASH[python_type] = hash_fn

# Create slang marshall for reflection type


SLANG_MARSHALS_BY_FULL_NAME: dict[str, type[BaseSlangTypeMarshal]] = {}
SLANG_MARSHALS_BY_NAME: dict[str, type[BaseSlangTypeMarshal]] = {}
SLANG_MARSHALS_BY_KIND: dict[TypeReflection.Kind, type[BaseSlangTypeMarshal]] = {
    TypeReflection.Kind.scalar: ScalarSlangTypeMarshal,
    TypeReflection.Kind.vector: VectorSlangTypeMarshal,
    TypeReflection.Kind.matrix: MatrixSlangTypeMarshal,
    TypeReflection.Kind.struct: StructSlangTypeMarshal,
}


def create_slang_type_marshal(slang_type: TypeReflection) -> BaseSlangTypeMarshal:
    marshal = SLANG_MARSHALS_BY_FULL_NAME.get(slang_type.full_name, None)
    if marshal is not None:
        return marshal(slang_type)
    marshal = SLANG_MARSHALS_BY_NAME.get(slang_type.name, None)
    if marshal is not None:
        return marshal(slang_type)
    marshal = SLANG_MARSHALS_BY_KIND.get(slang_type.kind, None)
    if marshal is not None:
        return marshal(slang_type)
    raise ValueError(f"Unsupported slang type {slang_type.full_name}")

# Node in a built signature tree, maintains a pairing of python+slang marshall,
# and a potential set of child nodes


class SignatureNode:
    def __init__(self, value: Any):
        super().__init__()
        self.python_marshal = PYTHON_TYPE_MARSHAL[(type(value))](value)
        self.children: Optional[dict[str, SignatureNode]] = None
        if isinstance(value, dict):
            self.children = {x: SignatureNode(y) for x, y in value.items()}
        self.slang_marshall: Optional[BaseSlangTypeMarshal] = None
        self.param_index = -1
        self.type_shape: Optional[list[int]] = None
        self.argument_shape: Optional[list[Optional[int]]] = None
        self.transform_inputs: TConcreteOrUndefinedShape = None
        self.transform_outputs: TConcreteOrUndefinedShape = None
        self.call_transform: Optional[list[int]] = None

    # Check if compatible with a variable or function return type
    def is_compatible(
        self, slang_reflection: Union[VariableReflection, FunctionReflection]
    ) -> bool:
        slang_type = slang_reflection.type if isinstance(
            slang_reflection, VariableReflection) else slang_reflection.return_type
        if not self.python_marshal.is_compatible(slang_type):
            return False
        if self.children is not None:
            fields = slang_type.fields
            if len(fields) != len(self.children):
                return False
            fields_by_name = {x.name: x for x in slang_type.fields}
            for name, node in self.children.items():
                childfield = fields_by_name.get(name, None)
                if childfield is None:
                    return False
                if not node.is_compatible(childfield):
                    return False
        return True

    # Creates the slang marshall for the node, applies any options and
    # calculates the argument/type shapes
    def apply_signature(
        self,
        slang_reflection: Union[VariableReflection, FunctionReflection],
        path: str,
        input_transforms: Optional[dict[str, TConcreteShape]],
        output_transforms: Optional[dict[str, TConcreteShape]]
    ):
        slang_type = slang_reflection.type if isinstance(
            slang_reflection, VariableReflection) else slang_reflection.return_type
        self.slang_marshall = create_slang_type_marshal(slang_type)
        if self.children is not None:
            fields_by_name = {x.name: x for x in slang_type.fields}
            for name, node in self.children.items():
                node.apply_signature(
                    fields_by_name[name], f"{path}.{name}", input_transforms, output_transforms)

        if self.children is None:
            if input_transforms is not None:
                self.transform_inputs = input_transforms.get(path, self.transform_inputs)
            if output_transforms is not None:
                self.transform_outputs = output_transforms.get(
                    path, self.transform_outputs)
            self._calculate_argument_shape()

    # Recursively populate flat list of argument nodes

    def get_input_list(self, args: list['SignatureNode']):
        self._get_input_list_recurse(args)
        return args

    def _get_input_list_recurse(self, args: list['SignatureNode']):
        if self.children is not None:
            for child in self.children.values():
                child._get_input_list_recurse(args)
        if self.type_shape is not None:
            args.append(self)

    def write_call_data_pre_dispatch(self, call_data: dict[str, Any], value: Any):
        pass

    def read_call_data_post_dispatch(self, call_data: dict[str, Any], value: Any):
        pass

    def __repr__(self):
        return self.python_marshal.__repr__()

    # Verify / build concrete shaping
    # - where both are defined they must match
    # - where param is defined and input is not, set input to param
    # - where input is defined and param is not, set param to input
    # - if end up with undefined type shape, bail
    def _calculate_argument_shape(self):
        assert self.slang_marshall is not None
        input_shape = self.python_marshal.shape
        param_shape = self.slang_marshall.shape
        if input_shape is not None:
            type_len = len(param_shape)
            input_len = len(input_shape)
            type_end = type_len - 1
            input_end = input_len - 1
            new_param_type_shape: list[int] = []
            for i in range(type_len):
                param_dim_idx = type_end - i
                input_dim_idx = input_end - i
                param_dim_size = param_shape[param_dim_idx]
                input_dim_size = input_shape[input_dim_idx]
                if param_dim_size is not None and input_dim_size is not None:
                    if param_dim_size != input_dim_size:
                        raise ValueError(
                            f"Arg {self.param_index}, PS[{param_dim_idx}] != IS[{input_dim_idx}], {param_dim_size} != {input_dim_size}"
                        )
                    new_param_type_shape.append(param_dim_size)
                elif param_dim_size is not None:
                    new_param_type_shape.append(param_dim_size)
                elif input_dim_size is not None:
                    new_param_type_shape.append(input_dim_size)
                else:
                    raise ValueError(f"Arg {self.param_index} type shape is ambiguous")
            new_param_type_shape.reverse()
            self.type_shape = new_param_type_shape
            self.argument_shape = list(input_shape[: input_len - type_len])
        else:
            # If input not defined, parameter shape is the argument shape
            if None in param_shape:
                raise ValueError(f"Arg {self.param_index} type shape is ambiguous")
            self.type_shape = list(cast(TConcreteShape, param_shape))
            self.argument_shape = None

        if self.argument_shape is None:
            return

        # Verify transforms match argument shape
        if self.transform_inputs is not None and len(self.transform_inputs) == len(self.argument_shape):
            raise ValueError(
                f"Transform inputs {self.transform_inputs} must have the same number of dimensions as the argument shape {self.argument_shape}")
        if self.transform_outputs is not None and len(self.transform_outputs) == len(self.argument_shape):
            raise ValueError(
                f"Transform outputs {self.transform_outputs} must have the same number of dimensions as the argument shape {self.argument_shape}")

        # Define a default function transform which basically maps argument
        # dimensions to call dimensions 1-1, with a bit of extra work to handle
        # arguments that aren't the same size or shapes that aren't defined.
        # This is effectively what numpy does.
        self.call_transform = [i for i in range(len(self.argument_shape))]

        # Inject any custom transforms
        if self.transform_outputs is not None:
            for i in range(len(self.argument_shape)):
                if self.transform_outputs[i] is not None:
                    self.call_transform[i] = self.transform_outputs[i]


# Efficient function to build a sha256 hash that uniquiely identifies
# a function call signature (by types and shapes).
def build_signature_hash(*args: Any, **kwargs: Any) -> str:
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


# Internal function that walks the value tree to build a hashable
# buffer containing the types and shapes of the values.
def _recurse_build_signature_hash(stream: StringIO, python_value: Any):
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


# Builds the initial trees of signature nodes for a given set of
# python arguments and keyword arguments.
def build_signature(*args: Any, **kwargs: Any) -> TCallSignature:
    arg_signatures = [SignatureNode(x) for x in args]
    kwarg_signatures = {k: SignatureNode(v) for k, v in kwargs.items()}
    return (arg_signatures, kwarg_signatures)


# Tests if a signature is compatible with a slang function
def match_signature(
    signature: TCallSignature,
    function_reflection: FunctionReflection,
    call_mode: CallMode
) -> Union[None, TMatchedSignature]:
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
        matched_rval.slang_marshall = create_slang_type_marshal(
            function_reflection.return_type)

    if rval is not None:
        matched_params["_result"] = rval
    return matched_params  # type: ignore

# Add slang type marshals to the signature nodes for a function it has been matched to


def apply_signature(
        signature: TMatchedSignature,
        function_reflection: FunctionReflection,
        input_transforms: Optional[dict[str, TConcreteShape]] = None,
        output_transforms: Optional[dict[str, TConcreteShape]] = None):

    for name, node in signature.items():
        reflection = function_reflection if name == "_result" else function_reflection.parameters[
            node.param_index]
        node.apply_signature(reflection, name, input_transforms, output_transforms)

# Given the shapes of the parameters (inferred from reflection) and inputs (passed in by the user), calculates the
# argument shapes and call shape.
# All parameters must have a shape, however individual dimensions can have an undefined size (None)
# Inputs can also be fully undefined


def calculate_and_apply_call_shape(signature: TMatchedSignature) -> list[int]:
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
