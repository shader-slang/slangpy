# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, cast
import pytest
import deepdiff

import slangpy as spy
import slangpy.reflection as spyr
import slangpy.core.native as spyn
import slangpy.testing.helpers as helpers

from slangpy.reflection.typeresolution import resolve_function, ResolveResult

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
            "implicit_element_casts": True,
            "implicit_tensor_casts": True,
            "strict_broadcasting": False,
        },
        spyn.CallDataMode.global_data,
    )

    return context, spy.bindings.BoundCall(context, *unpacked_args, **unpacked_kwargs)


def resolve(
    bind_context: spy.bindings.BindContext,
    functions: list[spyr.SlangFunction],
    bindings: spy.bindings.BoundCall,
):
    all_functions: list[spyr.SlangFunction] = []
    for func in functions:
        if func.is_overloaded:
            all_functions.extend(func.overloads)
        else:
            all_functions.append(func)

    resolutions: list[ResolveResult] = []
    for func in all_functions:
        resolution = resolve_function(bind_context, func, bindings)
        if resolution:
            resolutions.append(resolution)

    return resolutions


def check(actual_resolution: ResolveResult, *args: spyr.SlangType, **kwargs: spyr.SlangType):
    """Check that the actual resolution matches the expected resolution."""
    if len(actual_resolution.args) != len(args):
        assert (
            False
        ), f"Argument count mismatch: expected {len(actual_resolution.args)}, got {len(args)}"
    for i in range(len(actual_resolution.args)):
        exp_type = actual_resolution.args[i]
        act_type = args[i]
        assert (
            exp_type.full_name == act_type.full_name
        ), f"Argument {i} type mismatch: expected {exp_type.full_name}, got {act_type.full_name}"

    if len(actual_resolution.kwargs) != len(kwargs):
        assert (
            False
        ), f"Keyword argument count mismatch: expected {len(actual_resolution.kwargs)}, got {len(kwargs)}"
    for name in actual_resolution.kwargs:
        assert name in kwargs, f"Keyword argument {name} missing in actual resolution"
        exp_type = actual_resolution.kwargs[name]
        act_type = kwargs[name]
        assert (
            exp_type.full_name == act_type.full_name
        ), f"Keyword argument {name} type mismatch: expected {exp_type.full_name}, got {act_type.full_name}"


def get_functions(module: spy.Module, names: str | list[str]) -> list[spyr.SlangFunction]:
    """Get one or more functions from a module by name."""
    functions = []
    if isinstance(names, str):
        names = [names]
    for name in names:
        func = module.layout.find_function_by_name(name)
        assert func is not None, f"Function {name} not found"
        assert func.name == name
        functions.append(func)
    return functions


def build_and_resolve(
    module: spy.Module, names: str | list[str], call_mode: spyn.CallMode, *args: Any, **kwargs: Any
):
    """Build test data and resolve the specified function(s) with the provided arguments."""
    functions = get_functions(module, names)
    context, bindings = build_test_data(module, call_mode, *args, **kwargs)
    resolutions = resolve(context, functions, bindings)
    assert len(resolutions) == 1, f"Expected one resolution, got {len(resolutions)}"
    return resolutions[0]


def get_type(module: spy.Module, name: str):
    """Get a type from a module by name."""
    t = module.layout.find_type_by_name(name)
    assert t is not None, f"Type {name} not found"
    return t


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_vectorize_type(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)
    module = helpers.create_module(
        device,
        """
import slangpy;

struct SimpleStruct {}
struct GenericStruct<T> {}

void testfunc<T>(GenericStruct<T> gs) {}
""",
    )

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
            assert (
                fnclean == expclean
            ), f"Expected specialization of {binding_type} to {marshall_type} to be {expected}, got {res.full_name}"
        else:
            assert not spyr.vectorize_type(marshall_type, binding_type, module.layout)

    check("int", "ValueType<int>", "int")
    check("int", "ValueType<bool>", "int")
    check("uint8_t", "ValueType<int>", "uint8_t")
    check("bool", "ValueType<int>", "bool")
    check("float", "ValueType<int>", "float")
    check("float", "ValueType<float>", "float")
    check("half", "ValueType<float>", "half")
    check("double", "ValueType<float>", "double")
    check("Unknown", "ValueType<int>", "int")
    check("SimpleStruct", "ValueType<SimpleStruct>", "SimpleStruct")
    check("GenericStruct<int>", "ValueType<GenericStruct<int>>", "GenericStruct<int>")

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
    check("half[3]", "Array1DValueType<float,3>", "half[3]")
    check("double[3]", "Array1DValueType<float,3>", "double[3]")
    check("half[0]", "Array1DValueType<float,3>", "half[3]")
    check("float[4]", "Array1DValueType<float,3>", None)
    check("int[3]", "Array1DValueType<float,3>", "int[3]")

    check(
        "StructuredBuffer<int>",
        "BufferMarshall<int,true>",
        "StructuredBuffer<int,DefaultDataLayout>",
    )
    check(
        "StructuredBuffer<int>",
        "BufferMarshall<int,false>",
        "StructuredBuffer<int,DefaultDataLayout>",
    )
    check("RWStructuredBuffer<int>", "BufferMarshall<int,false>", None)
    check(
        "RWStructuredBuffer<int>",
        "BufferMarshall<int,true>",
        "RWStructuredBuffer<int,DefaultDataLayout>",
    )
    check(
        "StructuredBuffer<Unknown>",
        "BufferMarshall<int,true>",
        "StructuredBuffer<int,DefaultDataLayout>",
    )
    check(
        "StructuredBuffer<Unknown>",
        "BufferMarshall<int,false>",
        "StructuredBuffer<int,DefaultDataLayout>",
    )
    check("RWStructuredBuffer<Unknown>", "BufferMarshall<int,false>", None)
    check(
        "RWStructuredBuffer<Unknown>",
        "BufferMarshall<int,true>",
        "RWStructuredBuffer<int,DefaultDataLayout>",
    )
    check("StructuredBuffer<float>", "BufferMarshall<int,true>", None)
    check("StructuredBuffer<float>", "BufferMarshall<int,false>", None)
    check("RWStructuredBuffer<float>", "BufferMarshall<int,false>", None)
    check("RWStructuredBuffer<float>", "BufferMarshall<int,true>", None)

    check("ByteAddressBuffer", "BufferMarshall<int,false>", "ByteAddressBuffer")
    check("RWByteAddressBuffer", "BufferMarshall<int,false>", None)
    check("ByteAddressBuffer", "BufferMarshall<int,true>", "ByteAddressBuffer")
    check("RWByteAddressBuffer", "BufferMarshall<int,true>", "RWByteAddressBuffer")

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
    check(
        "Ptr<int>",
        "NDBufferMarshall<int,1,true>",
        "Ptr<int, Access.ReadWrite, AddressSpace.Device>",
    )

    check(
        "StructuredBuffer<int>",
        "NDBufferMarshall<int,1,true>",
        "StructuredBuffer<int,DefaultDataLayout>",
    )
    check(
        "StructuredBuffer<int>",
        "NDBufferMarshall<int,1,false>",
        "StructuredBuffer<int,DefaultDataLayout>",
    )
    check("RWStructuredBuffer<int>", "NDBufferMarshall<int,1,false>", None)
    check(
        "RWStructuredBuffer<int>",
        "NDBufferMarshall<int,1,true>",
        "RWStructuredBuffer<int,DefaultDataLayout>",
    )
    check(
        "RWStructuredBuffer<int>",
        "NDBufferMarshall<int,1,1>",
        "RWStructuredBuffer<int,DefaultDataLayout>",
    )
    check(
        "StructuredBuffer<Unknown>",
        "NDBufferMarshall<int,1,true>",
        "StructuredBuffer<int,DefaultDataLayout>",
    )
    check(
        "StructuredBuffer<Unknown>",
        "NDBufferMarshall<int,1,false>",
        "StructuredBuffer<int,DefaultDataLayout>",
    )
    check("RWStructuredBuffer<Unknown>", "NDBufferMarshall<int,1,false>", None)
    check(
        "RWStructuredBuffer<Unknown>",
        "NDBufferMarshall<int,1,true>",
        "RWStructuredBuffer<int,DefaultDataLayout>",
    )
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

    # This is an ambiguous case that could resolve to int[10] (i.e. the element type) or int[10][10] (i.e. loading array)
    # In practice, it needs resolving python side. Vectors/matrices don't suffer from this issue as it is not possible to have
    # a vector/matrix of vectors/matrices in slang.
    check("Unknown[10]", "NDBufferMarshall<int[10],1,false>", None)
    check("Unknown[0]", "NDBufferMarshall<int[10],1,false>", None)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_no_parameters(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)
    module = helpers.create_module(
        device,
        """
void test_func() {}
""",
    )

    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim)
    check(actual_resolution)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
def test_simple_value(device_type: spy.DeviceType, param_count: int):
    device = helpers.get_device(type=device_type)
    params = ", ".join([f"int p{i}" for i in range(param_count)])
    module = helpers.create_module(
        device,
        f"""
void test_func({params}) {{}}
""",
    )
    param_values = (0,) * param_count
    param_types = (get_type(module, "int"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
def test_generic_value(device_type: spy.DeviceType, param_count: int):
    device = helpers.get_device(type=device_type)
    params = ", ".join([f"T p{i}" for i in range(param_count)])
    module = helpers.create_module(
        device,
        f"""
void test_func<T>({params}) {{}}
""",
    )
    param_values = (0,) * param_count
    param_types = (get_type(module, "int"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
def test_simple_vector_value(device_type: spy.DeviceType, param_count: int):
    device = helpers.get_device(type=device_type)
    params = ", ".join([f"vector<int,1> p{i}" for i in range(param_count)])
    module = helpers.create_module(
        device,
        f"""
void test_func({params}) {{}}
""",
    )
    param_values = (spy.int1(0),) * param_count
    param_types = (get_type(module, "vector<int,1>"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic_type", [False, True])
@pytest.mark.parametrize("generic_count", [False, True])
def test_vector_value(
    device_type: spy.DeviceType, param_count: int, generic_type: bool, generic_count: bool
):
    device = helpers.get_device(type=device_type)

    if generic_type and generic_count:
        sig_generic = "<T, let N: int>"
        sig_param_type = "vector<T,N>"
    elif generic_type:
        sig_generic = "<T>"
        sig_param_type = "vector<T,1>"
    elif generic_count:
        sig_generic = "<let N: int>"
        sig_param_type = "vector<int,N>"
    else:
        sig_generic = ""
        sig_param_type = "vector<int,1>"

    params = ", ".join([f"{sig_param_type} p{i}" for i in range(param_count)])
    module = helpers.create_module(
        device,
        f"""
void test_func{sig_generic}({params}) {{}}
""",
    )
    param_values = (spy.int1(0),) * param_count
    param_types = (get_type(module, "vector<int,1>"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


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
    check(actual_resolution, *param_types)


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
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
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
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
def test_ndbuffer_vectorize_vectortype(
    device_type: spy.DeviceType, param_count: int, generic: bool
):
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
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
def test_ndbuffer_vectorize_arraytype(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "int[1]" if not generic else "T[1]"
    sig_params = ", ".join([f"{sig_param_type} p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.NDBuffer.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "int[1]"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
def test_ndbufferarray_vectorize_arraytype(
    device_type: spy.DeviceType, param_count: int, generic: bool
):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T: __BuiltinIntegerType>"
    sig_param_type = "int[1]" if not generic else "T[1]"
    sig_params = ", ".join([f"{sig_param_type} p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (
        spy.NDBuffer.empty(device, (10,), dtype=get_type(module, "int[1]")),
    ) * param_count
    param_types = (get_type(module, "int[1]"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [True])  # Only generic makes sense here
def test_ndbufferarray_vectorize_arraytype2(
    device_type: spy.DeviceType, param_count: int, generic: bool
):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T: IArray<int>>"
    sig_param_type = "int[1]" if not generic else "T[1]"
    sig_params = ", ".join([f"{sig_param_type} p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (
        spy.NDBuffer.empty(device, (10,), dtype=get_type(module, "int[1]")),
    ) * param_count
    param_types = (get_type(module, "int[1][1]"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_typedef(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    code = f"""
import slangpy;
typedef int Foo;
void test_func(Foo x) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.NDBuffer.empty(device, (10,), dtype=get_type(module, "int")),)
    param_types = (get_type(module, "int"),)
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_interface(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    code = f"""
import slangpy;

interface IFoo {{}}

extension int: IFoo {{}}

void test_func(IFoo x) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (0,)
    param_types = (get_type(module, "int"),)
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_interface_generic(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    code = f"""
import slangpy;

interface IFoo {{}}

extension int: IFoo {{}}

void test_func<T: IFoo>(T x) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (0,)
    param_types = (get_type(module, "int"),)
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_ndbufferarray_generic_struct(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    code = f"""
import slangpy;

struct Foo<T> {{ T value; }}

void test_func<T>(Foo<T> p0) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.NDBuffer.empty(device, (10,), dtype=get_type(module, "Foo<int>")),)
    param_types = (get_type(module, "Foo<int>"),)
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)
