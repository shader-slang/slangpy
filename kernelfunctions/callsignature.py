from enum import Enum
import hashlib
from io import StringIO
from typing import Any, Callable, Optional, Union
from sgl import FunctionReflection, TypeReflection, VariableReflection


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

    def __repr__(self):
        return self.name


# Base class for marshalling python types
class BasePythonTypeMarshal:
    def __init__(self, python_type: type):
        super().__init__()
        self.type = python_type

    def is_compatible(self, slang_type: TypeReflection) -> bool:
        raise NotImplementedError()

    @property
    def name(self):
        return self.type.__name__

    def __repr__(self):
        return self.type.__name__


# Dictionary of python types to corresponding marshall
PYTHON_TYPE_MARSHAL: dict[type, BasePythonTypeMarshal] = {}

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
    marshall: BasePythonTypeMarshal,
    hash_fn: Optional[Callable[[StringIO, Any], Any]],
):
    PYTHON_TYPE_MARSHAL[python_type] = marshall
    PYTHON_SIGNATURE_HASH[python_type] = hash_fn

# Create slang marshall for reflection type


def create_slang_type_marshal(slang_type: TypeReflection) -> BaseSlangTypeMarshal:
    return BaseSlangTypeMarshal(slang_type)

# Node in a built signature tree, maintains a pairing of python+slang marshall,
# and a potential set of child nodes


class SignatureNode:
    def __init__(self, value: Any):
        super().__init__()
        self.python_marshal = PYTHON_TYPE_MARSHAL[(type(value))]
        self.children: Optional[dict[str, SignatureNode]] = None
        if isinstance(value, dict):
            self.children = {x: SignatureNode(y) for x, y in value.items()}
        self.slang_marshall: Optional[BaseSlangTypeMarshal] = None
        self.param_index = -1

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

    # Creates the slang marshall for the node
    def populate_slang_types(
        self, slang_reflection: Union[VariableReflection, FunctionReflection]
    ):
        slang_type = slang_reflection.type if isinstance(
            slang_reflection, VariableReflection) else slang_reflection.return_type
        self.slang_marshall = BaseSlangTypeMarshal(slang_type)
        if self.children is not None:
            fields_by_name = {x.name: x for x in slang_type.fields}
            for name, node in self.children.items():
                node.populate_slang_types(fields_by_name[name])

    def write_call_data_pre_dispatch(self, call_data: dict[str, Any], value: Any):
        pass

    def read_call_data_post_dispatch(self, call_data: dict[str, Any], value: Any):
        pass

    def __repr__(self):
        return self.python_marshal.__repr__()


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

    if rval is not None:
        matched_params["_result"] = rval
    return matched_params  # type: ignore

# Add slang type marshals to the signature nodes for a function it has been matched to


def populate_slang_types(signature: TMatchedSignature, function_reflection: FunctionReflection):
    for name, node in signature.items():
        if name == "_result":
            node.populate_slang_types(function_reflection)
        else:
            node.populate_slang_types(function_reflection.parameters[node.param_index])


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
