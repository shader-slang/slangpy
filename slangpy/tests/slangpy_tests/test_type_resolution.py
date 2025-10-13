from typing import Any, Optional, cast
import pytest
import deepdiff

import slangpy as spy
import slangpy.reflection as spyr
import slangpy.core.native as spyn
import slangpy.testing.helpers as helpers

# Limit the device types tested here to just one, since these tests are not device specific.
DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES[0:1]

def build_test_data(module: spy.Module, call_mode: spyn.CallMode, *args: Any, **kwargs: Any):
    """
    Very cut down version of the first parts of CallData construction, in which user provided arguments
    are inpacked and the bound call information is constructed, at this stage with no knowledge of the
    slang function being called."""

    # Build 'unpacked' args (that handle IThis) and extract any pytorch
    # tensor references at the same time.
    tensor_refs = []
    unpacked_args = spyn.unpack_refs_and_args(tensor_refs, *args)
    unpacked_kwargs = spyn.unpack_refs_and_kwargs(tensor_refs, **kwargs)

    # Setup context
    context = spy.bindings.BindContext(
        module.layout,
        call_mode,
        module.device_module,
        {
            "implicit_element_casts" : True,
            "implicit_tensor_casts" : True,
            "strict_broadcasting" : False,
        },
        spyn.CallDataMode.global_data,
    )

    return context, spy.bindings.BoundCall(context, *unpacked_args, **unpacked_kwargs)


class ResolveResult:
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

TResolutionArgs = tuple[spyn.NativeMarshall, spyr.SlangType, spyr.SlangType|None]

def resolve_arguments(bind_context: spy.bindings.BindContext, args: list[TResolutionArgs]):
    """ Check if the provided arguments can be converted to the target types. """
    for i in range(len(args)):
        marshall, provided_type, target_type = args[i]
        if target_type:
            continue

        if hasattr(marshall, "resolve_types"):
            types = marshall.resolve_types(bind_context, provided_type)
        else:
            types = [marshall.resolve_type(bind_context, provided_type)]

        resolved_args = []
        for potential_type in types:
            new_args = args.copy()
            new_args[i] = (marshall, provided_type, potential_type)
            new_args = resolve_arguments(bind_context, new_args)
            if new_args:
                resolved_args.append(new_args)

        if len(resolved_args) == 1:
            return resolved_args[0]
            continue
        else:
            return None

    return args

def resolve_function(bind_context: spy.bindings.BindContext, function: spyr.SlangFunction, call_mode: spyn.CallMode, bindings: spy.bindings.BoundCall):
    assert not function.is_overloaded
    function_parameters = [x for x in function.parameters]
    signature_args = bindings.args
    signature_kwargs = bindings.kwargs

    # Build empty positional list of python arguments to correspond to each slang argument
    positioned_args: list[spy.bindings.BoundVariable|None] = [None] * len(function_parameters)

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

    # If we reach this point, all arguments are successfully bound
    current_args: list[tuple[spy.bindings.Marshall, spyr.SlangType, spyr.SlangType|None]] = \
        [(cast(spy.bindings.Marshall,param.python), arg.type, None) for param, arg in zip(positioned_args, function_parameters)]
    resolved_args = resolve_arguments(bind_context, current_args)

    res = ResolveResult()

    pos_arg_types = []
    for i,arg in enumerate(resolved_args):
        marshall, provided_type, target_type = arg
        pos_arg_types.append(target_type)
    res.args = tuple(pos_arg_types)

    for name, arg in signature_kwargs.items():
        if name == "_result":
            continue
        i = name_map[name]
        marshall, provided_type, target_type = resolved_args[i]
        res.kwargs[name] = target_type

    return res

def resolve(bind_context: spy.bindings.BindContext, functions: list[spyr.SlangFunction], call_mode: spyn.CallMode, bindings: spy.bindings.BoundCall):
    all_functions: list[spyr.SlangFunction] = []
    for func in functions:
        if func.is_overloaded:
            all_functions.extend(func.overloads)
        else:
            all_functions.append(func)

    resolutions: list[ResolveResult] = []
    for func in all_functions:
        resolution = resolve_function(bind_context, func, call_mode, bindings)
        if resolution:
            resolutions.append(resolution)

    return resolutions


def check(expected_resolution: ResolveResult, actual_resolution: ResolveResult):
    """ Check that the actual resolution matches the expected resolution. """
    diff = deepdiff.DeepDiff(expected_resolution, actual_resolution)
    assert diff == {}, f"Resolution mismatch: {diff}"


def get_functions(module: spy.Module, names: str|list[str]) -> list[spyr.SlangFunction]:
    """ Get one or more functions from a module by name. """
    functions = []
    if isinstance(names, str):
        names = [names]
    for name in names:
        func = module.layout.find_function_by_name(name)
        assert func is not None, f"Function {name} not found"
        assert func.name == name
        functions.append(func)
    return functions

def build_and_resolve(module: spy.Module, names: str|list[str], call_mode: spyn.CallMode, *args: Any, **kwargs: Any):
    """ Build test data and resolve the specified function(s) with the provided arguments. """
    functions = get_functions(module, names)
    context, bindings = build_test_data(module, call_mode, *args, **kwargs)
    resolutions = resolve(context, functions, call_mode, bindings)
    assert len(resolutions) == 1, f"Expected one resolution, got {len(resolutions)}"
    return resolutions[0]

def get_type(module: spy.Module, name: str):
    """ Get a type from a module by name. """
    t = module.layout.find_type_by_name(name)
    assert t is not None, f"Type {name} not found"
    return t

@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_vectorize_type(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)
    module = helpers.create_module(device, """
import slangpy;
""")

    def check(binding_type: str, marshall_type: str, expected: Optional[str]):
        bt = module.layout.find_type_by_name(binding_type)
        assert bt is not None, f"Could not find type {binding_type}"
        mt = module.layout.find_type_by_name(marshall_type)
        assert mt is not None, f"Could not find type {marshall_type}"
        if expected:
            res = spyr.vectorize_type(marshall_type, binding_type, module.layout)
            assert res is not None, f"Expected specialization of {binding_type} to {marshall_type}"
            fnclean = res.full_name.replace(" ", "")
            expclean = expected.replace(" ", "")
            assert fnclean == expclean, f"Expected specialization of {binding_type} to {marshall_type} to be {expected}, got {res.full_name}"
        else:
            assert not spyr.vectorize_type(marshall_type, binding_type, module.layout)

    check("int", "ValueType<int>", "int")
    check("float", "ValueType<int>", None)
    check("Unknown", "ValueType<int>", None) # Fully generics not handled

    check("int", "VectorValueType<int,1>", "int")
    check("float", "VectorValueType<int,1>", None)
    check("vector<int,1>", "VectorValueType<int,1>", "vector<int,1>")
    check("vector<int,0>", "VectorValueType<int,1>", "vector<int,1>")
    check("vector<Unknown,1>", "VectorValueType<int,1>", "vector<int,1>")
    check("vector<Unknown,0>", "VectorValueType<int,1>", "vector<int,1>")
    check("float3", "VectorValueType<float,3>", "vector<float,3>")
    check("float4", "VectorValueType<float,3>", None)
    check("int3", "VectorValueType<float,3>", None)

    check("int", "Array1DValueType<int,1>", "int")
    check("float", "Array1DValueType<int,1>", None)
    check("vector<int,1>", "Array1DValueType<int,1>", None)
    check("int[1]", "Array1DValueType<int,1>", "int[1]")
    check("Unknown[1]", "Array1DValueType<int,1>", "int[1]")
    check("int[0]", "Array1DValueType<int,1>", "int[1]")
    check("Unknown[0]", "Array1DValueType<int,1>", "int[1]")
    check("float[3]", "Array1DValueType<float,3>", "float[3]")
    check("float[4]", "Array1DValueType<float,3>", None)
    check("int[3]", "Array1DValueType<float,3>", None)



    check("NDBuffer<int,1>", "NDBufferMarshall<int,1,true>", "NDBuffer<int,1>")
    check("NDBuffer<int,1>", "NDBufferMarshall<int,1,false>", "NDBuffer<int,1>")
    check("RWNDBuffer<int,1>", "NDBufferMarshall<int,1,false>", None)
    check("RWNDBuffer<int,1>", "NDBufferMarshall<int,1,true>", "RWNDBuffer<int,1>")
    check("NDBuffer<Unknown,1>", "NDBufferMarshall<int,1,true>", "NDBuffer<int,1>")
    check("NDBuffer<Unknown,1>", "NDBufferMarshall<int,1,false>", "NDBuffer<int,1>")
    check("RWNDBuffer<Unknown,1>", "NDBufferMarshall<int,1,false>", None)
    check("RWNDBuffer<Unknown,1>", "NDBufferMarshall<int,1,true>", "RWNDBuffer<int,1>")
    check("NDBuffer<float,1>", "NDBufferMarshall<int,1,true>", None)
    check("NDBuffer<float,1>", "NDBufferMarshall<int,1,false>", None)
    check("RWNDBuffer<float,1>", "NDBufferMarshall<int,1,false>", None)
    check("RWNDBuffer<float,1>", "NDBufferMarshall<int,1,true>", None)

    check("StructuredBuffer<int>", "NDBufferMarshall<int,1,true>", "StructuredBuffer<int,DefaultDataLayout>")
    check("StructuredBuffer<int>", "NDBufferMarshall<int,1,false>", "StructuredBuffer<int,DefaultDataLayout>")
    check("RWStructuredBuffer<int>", "NDBufferMarshall<int,1,false>", None)
    check("RWStructuredBuffer<int>", "NDBufferMarshall<int,1,true>", "RWStructuredBuffer<int,DefaultDataLayout>")
    check("StructuredBuffer<Unknown>", "NDBufferMarshall<int,1,true>", "StructuredBuffer<int,DefaultDataLayout>")
    check("StructuredBuffer<Unknown>", "NDBufferMarshall<int,1,false>", "StructuredBuffer<int,DefaultDataLayout>")
    check("RWStructuredBuffer<Unknown>", "NDBufferMarshall<int,1,false>", None)
    check("RWStructuredBuffer<Unknown>", "NDBufferMarshall<int,1,true>", "RWStructuredBuffer<int,DefaultDataLayout>")
    check("StructuredBuffer<float>", "NDBufferMarshall<int,1,true>", None)
    check("StructuredBuffer<float>", "NDBufferMarshall<int,1,false>", None)
    check("RWStructuredBuffer<float>", "NDBufferMarshall<int,1,false>", None)
    check("RWStructuredBuffer<float>", "NDBufferMarshall<int,1,true>", None)

    check("int", "NDBufferMarshall<int,1,true>", "int")
    check("float", "NDBufferMarshall<int,1,true>", None)
    check("int", "NDBufferMarshall<int,1,false>", "int")
    check("float", "NDBufferMarshall<int,1,false>", None)
    check("vector<int,0>", "NDBufferMarshall<int,1,false>", "vector<int,0>")
    check("vector<int,3>", "NDBufferMarshall<int,1,false>", "vector<int,3>")
    check("int3", "NDBufferMarshall<int,1,false>", "vector<int,3>")
    check("vector<float,1>", "NDBufferMarshall<int,1,false>", None)
    check("matrix<int,3,3>", "NDBufferMarshall<int,1,false>", "matrix<int,3,3>")
    check("int[10]", "NDBufferMarshall<int[10],1,false>", "int[10]")
    check("int[0]", "NDBufferMarshall<int[10],1,false>", "int[10]")



@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_no_parameters(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)
    module = helpers.create_module(device, """
void test_func() {}
""")

    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim)
    expected_resolution = ResolveResult()
    check(expected_resolution, actual_resolution)

@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
def test_simple_value(device_type: spy.DeviceType, param_count: int):
    device = helpers.get_device(type=device_type)
    params = ", ".join([f"int p{i}" for i in range(param_count)])
    module = helpers.create_module(device, f"""
void test_func({params}) {{}}
""")
    param_values = (0,) * param_count
    param_types = (get_type(module, "int"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    expected_resolution = ResolveResult(*param_types)
    check(expected_resolution, actual_resolution)

@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
def test_ndbuffer_none_vectorize(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "int" if not generic else "T"
    sig_params = ", ".join([f"NDBuffer<{sig_param_type},1> p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.NDBuffer.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "NDBuffer<int,1>"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    expected_resolution = ResolveResult(*param_types)
    check(expected_resolution, actual_resolution)

@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
def test_ndbuffer_structured_buffer(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "int" if not generic else "T"
    sig_params = ", ".join([f"StructuredBuffer<{sig_param_type}> p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.NDBuffer.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "StructuredBuffer<int>"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    expected_resolution = ResolveResult(*param_types)
    check(expected_resolution, actual_resolution)

@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False]) # Can't handle fully generic function as ambiguous
def test_ndbuffer_vectorize(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "int" if not generic else "T"
    sig_params = ", ".join([f"{sig_param_type} p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.NDBuffer.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "int"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    expected_resolution = ResolveResult(*param_types)
    check(expected_resolution, actual_resolution)

@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False,True])
def test_ndbuffer_vectorize_vectortype(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "vector<int,1>" if not generic else "vector<T,1>"
    sig_params = ", ".join([f"{sig_param_type} p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.NDBuffer.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "vector<int,1>"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    expected_resolution = ResolveResult(*param_types)
    check(expected_resolution, actual_resolution)
