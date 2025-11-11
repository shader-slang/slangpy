# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, cast
import pytest
import deepdiff

import slangpy as spy
import slangpy.reflection as spyr
import slangpy.core.native as spyn
import slangpy.testing.helpers as helpers

from slangpy.reflection.typeresolution import resolve_function, ResolveResult, ResolutionDiagnostic

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
            "strict_broadcasting": False,
        },
        spyn.CallDataMode.global_data,
    )

    return context, spy.bindings.BoundCall(context, *unpacked_args, **unpacked_kwargs)


def resolve(
    bind_context: spy.bindings.BindContext,
    functions: list[spyr.SlangFunction],
    bindings: spy.bindings.BoundCall,
    diagnostics: ResolutionDiagnostic,
):
    all_functions: list[spyr.SlangFunction] = []
    for func in functions:
        if func.is_overloaded:
            all_functions.extend(func.overloads)
        else:
            all_functions.append(func)

    resolutions: list[ResolveResult] = []
    for func in all_functions:
        resolution = resolve_function(bind_context, func, bindings, diagnostics)
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
        act_type = actual_resolution.args[i]
        exp_type = args[i]
        assert (
            exp_type.full_name == act_type.full_name
        ), f"Argument {i} type mismatch: expected {exp_type.full_name}, got {act_type.full_name}"

    if len(actual_resolution.kwargs) != len(kwargs):
        assert (
            False
        ), f"Keyword argument count mismatch: expected {len(actual_resolution.kwargs)}, got {len(kwargs)}"
    for name in actual_resolution.kwargs:
        assert name in kwargs, f"Keyword argument {name} missing in actual resolution"
        act_type = actual_resolution.kwargs[name]
        exp_type = kwargs[name]
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
    diagnostics = ResolutionDiagnostic()
    resolutions = resolve(context, functions, bindings, diagnostics)
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
    check("vector<int,0>", "NDBufferMarshall<int,1,false>", None)
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
@pytest.mark.parametrize("generic", [False])  # Fully generic not supported
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
    # Use unique device due to https://github.com/shader-slang/slang/issues/8954
    device = helpers.get_device(type=device_type, use_cache=False)

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
    # Use unique device due to https://github.com/shader-slang/slang/issues/8954
    device = helpers.get_device(type=device_type, use_cache=False)

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


def calc_usage(rw: bool) -> spy.BufferUsage:
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    return usage


class _Buffer:
    def __init__(self, element_count: int, struct_size: int, rw: bool):
        super().__init__()
        self.element_count = element_count
        self.struct_size = struct_size
        self.rw = rw

    def __repr__(self) -> str:
        return f"{'RW' if self.rw else ''}Buffer"

    def __call__(self, module: spy.Module):
        return module.device.create_buffer(
            element_count=self.element_count,
            struct_size=self.struct_size,
            usage=calc_usage(self.rw),
        )


class _NDBuffer:
    def __init__(self, base_type: str, dim: int, rw: bool):
        super().__init__()
        self.base_type = base_type
        self.dim = dim
        self.rw = rw

    def __repr__(self) -> str:
        return f"{'RW' if self.rw else ''}NDBuffer<{self.base_type},{self.dim}>)"

    def __call__(self, module: spy.Module) -> spy.NDBuffer:
        return spy.NDBuffer.empty(
            module.device,
            (3,) * self.dim,
            dtype=module.layout.find_type_by_name(self.base_type),
            usage=calc_usage(self.rw),
        )


class _Tensor:
    def __init__(self, base_type: str, dim: int, rw: bool):
        super().__init__()
        self.base_type = base_type
        self.dim = dim
        self.rw = rw

    def __repr__(self) -> str:
        return f"{'RW' if self.rw else ''}Tensor<{self.base_type},{self.dim}>)"

    def __call__(self, module: spy.Module) -> spy.Tensor:
        return spy.Tensor.empty(
            module.device,
            (3,) * self.dim,
            dtype=module.layout.find_type_by_name(self.base_type),
            usage=calc_usage(self.rw),
        )


# fmt: off

# List of simple tests, where each is a function, input argument value (or factor func + args), and expected resolved type / dimensionality.
# Functions are in type_resolution.slang
TESTS = [
    # Standard scalar float tests
    ("func_float", 0, "float", 0),
    ("func_float", 1.5, "float", 0),
    ("func_float", spy.float1(2.5), "float", 1),
    ("func_float", spy.float3(3.5, 4.5, 5.5), "float", 3),
    ("func_float", _NDBuffer("float", 1, False), "float", 1),
    ("func_float", _NDBuffer("float", 2, True), "float", 2),
    ("func_float", _Tensor("float", 1, False), "float", 1),
    ("func_float", _Tensor("float", 2, True), "float", 2),
    ("func_float",[1.5], "float", 1),
    ("func_float",[1], "float", 1),

    # These should fail as we don't implicit cast AND vectorize - its one or the other
    ("func_float", spy.int3(0,0,0), None, None),
    ("func_float", _NDBuffer("int", 1, False), None, None),
    ("func_float", _Tensor("int", 1, False), None, None),

    # Standard scalar int tests
    ("func_int", 42, "int", 0),
    ("func_int", 1.5, "int", 0), # Passes, as invalid fractional value has to be detected at runtime
    ("func_int", spy.int1(2), "int", 1),
    ("func_int", _NDBuffer("int", 1, False), "int", 1),

    # None default numeric types should be picked up correctly
    ("func_half", 1.0, "half", 0),
    ("func_half", 2, "half", 0),
    ("func_int8", 255, "int8_t", 0),
    ("func_int64", 2**64 - 1, "int64_t", 0),

    # Buffer/tensor -> function
    ("func_int64", _NDBuffer("int64_t", 2, True), "int64_t", 2),
    ("func_half", _Tensor("half", 1, False), "half", 1),

    # Fully generic, only unambiguous for types that can't be vectorized
    ("func_generic", 3.5, "float", 0),
    ("func_generic", 42, "int", 0),
    ("func_generic", spy.float1(2.5), None, None),
    ("func_generic", [42], "int", 0),
    ("func_generic", _NDBuffer("int", 3, False), None, None),
    ("func_generic", _Tensor("float", 2, True), None, None),

    # Generic constrained to built in integer allows vectorization
    ("func_genericint", 42, "int", 0),
    ("func_genericint", spy.int3(1,2,3), "int", 3),
    ("func_genericint", _NDBuffer("int", 2, False), "int", 2),

    # Same for floating point
    ("func_genericfloat", 42.5, "float", 0),
    ("func_genericfloat", spy.float3(1,2,3), "float", 3),
    ("func_genericfloat", _NDBuffer("float", 2, False), "float", 2),
    ("func_genericfloat", _Tensor("float", 2, False), "float", 2),

    # Also ensure specialization fails when types don't match
    ("func_genericint", _NDBuffer("float", 2, False), None, None),
    ("func_genericint", _Tensor("float", 2, False), None, None),

    # float3 tests
    ("func_float3", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_float3", _NDBuffer("float", 2, False), "vector<float,3>", 1),
    ("func_float3", _Tensor("float", 2, True), "vector<float,3>", 1),
    ("func_float3", _NDBuffer("float3", 1, False), "vector<float,3>", 1),
    ("func_float3", _Tensor("float3", 1, True), "vector<float,3>", 1),
    ("func_float3", _Tensor("float4", 1, True), None, None),

    # slang arg defined as vector<float,3> to ensure identical resolution
    ("func_vector_float3", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_vector_float3", _NDBuffer("float", 2, False), "vector<float,3>", 1),
    ("func_vector_float3", _Tensor("float", 2, True), "vector<float,3>", 1),

    # Other vector types
    ("func_int3", spy.int3(1, 2, 3), "vector<int,3>", 3),
    ("func_int3", _NDBuffer("int", 2, False), "vector<int,3>", 1),
    ("func_half3", _Tensor("half3", 1, True), "vector<half,3>", 1),

    # We should potentially allow these implicit casts as there is no vectorizing
    # happening, but for now we don't.
    ("func_float3", 1.0, None, None),
    ("func_half3", spy.float3(1.0, 2.0, 3.0), None, None),

    # generic typed vector tests
    ("func_vector3_generic", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_vector3_generic", _NDBuffer("float", 2, False), "vector<float,3>", 1),
    ("func_vector3_generic", _Tensor("float", 2, True), "vector<float,3>", 1),
    ("func_vector3_generic", _NDBuffer("float3", 1, False), "vector<float,3>", 1),
    ("func_vector3_generic", _Tensor("float3", 1, True), "vector<float,3>", 1),
    ("func_vector3_generic", _Tensor("float4", 1, True), None, None),

    # generic typed+dim vector tests
    ("func_vectorN_generic", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_vectorN_generic", _NDBuffer("float", 2, False), None, None),
    ("func_vectorN_generic", _Tensor("float", 2, True), None, None),
    ("func_vectorN_generic", _NDBuffer("float3", 1, False), "vector<float,3>", 1),
    ("func_vectorN_generic", _Tensor("float3", 1, True), "vector<float,3>", 1),
    ("func_vectorN_generic", _Tensor("float4", 1, True), "vector<float,4>", 1),

    # generic dim float tests (similar, but verify incorrect element type fails)
    ("func_floatN_generic", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_floatN_generic", _NDBuffer("float", 2, False), None, None),
    ("func_floatN_generic", _Tensor("float", 2, True), None, None),
    ("func_floatN_generic", _NDBuffer("float3", 1, False), "vector<float,3>", 1),
    ("func_floatN_generic", _Tensor("float3", 1, True), "vector<float,3>", 1),
    ("func_floatN_generic", _NDBuffer("int3", 1, False), None, None),
    ("func_floatN_generic", _Tensor("float4", 1, True), "vector<float,4>", 1),
    ("func_floatN_generic", _NDBuffer("int4", 1, False), None, None),

    # Basic float arrays
    ("func_float_array", [1,2,3,4], "float[4]", 3),
    ("func_float_array", _NDBuffer("float[4]",1,True), "float[4]", 3),
    ("func_float_array", _Tensor("float[4]",1,False), "float[4]", 3),
    ("func_float_array2", [1,2,3,4,5,6,7,8], "float[8]", 3),
    ("func_float_array2", _NDBuffer("float[8]",1,True), "float[8]", 3),
    ("func_float_array2", _Tensor("float[8]",1,False), "float[8]", 3),

    # Incorrect sizes/types
    ("func_float_array2", [1,2,3,4], None, None),
    ("func_float_array2", _NDBuffer("int[8]",1,True), None, None),

    # Unsized / generic arrays
    ("func_float_unsized_array", [1,2,3,4], "float[4]", 3),
    ("func_generic_array", [1,2,3,4], "int[4]", 3),
    ("func_generic_array", _NDBuffer("float[4]",1,False), "float[4]", 3),
    ("func_generic_array", _Tensor("float[4]",1,False), "float[4]", 3),
    ("func_generic_type_array", [1.5,2.5,3.5,4.6], "float[4]", 3),
    ("func_generic_length_array", [1.5,2.5,3.5,4.6], "float[4]", 3),
    ("func_generic_length_array", _NDBuffer("float[4]",1,False), "float[4]", 3),
    ("func_generic_length_array", _Tensor("float[4]",1,False), "float[4]", 3),
    ("func_generic_unsized_array", [1.5,2.5,3.5,4.6], "float[4]", 3),
    ("func_generic_unsized_array", _NDBuffer("float[4]",1,False), "float[4]", 3),
    ("func_generic_unsized_array", _Tensor("float[4]",1,False), "float[4]", 3),

    # These are ambiguous, as the function has T[4], so could resolve with T==float
    # or T==float[4].
    ("func_generic_type_array", _NDBuffer("float[4]",1,False), None, None),
    ("func_generic_type_array", _Tensor("float[4]",1,False), None, None),

    # Although dimension doesn't match (the function is T[4]), the fact that the
    # array is completely generic means this can resolve with T==float[8]
    ("func_generic_type_array", _NDBuffer("float[8]",1,False), "float[8][4]", 1),
    ("func_generic_type_array", _Tensor("float[8]",1,False), "float[8][4]", 1),

    # The constrained generic type (to __BuiltinFloatingPointType) means that
    # T==float is the only valid resolution, so it is no longer ambiguous
    ("func_generic_constrained_type_array", _NDBuffer("float[4]",1,False), "float[4]", 1),
    ("func_generic_constrained_type_array", _Tensor("float[4]",1,False), "float[4]", 1),

    # Generic array resolutions that should fail due to wrong size/type
    ("func_generic_length_array", _NDBuffer("int[4]",1,False), None, None),

    # Loading from container of floats to array of floats. As size can not
    # be determined at compile time, none finite length arrays can't be resolved.
    ("func_float_array", _NDBuffer("float",1,True), "float[4]", 3),
    ("func_float_array2", _NDBuffer("float",1,True), "float[8]", 3),
    ("func_float_unsized_array", _NDBuffer("float",1,False), None, None),
    ("func_generic_array", _NDBuffer("float",1,False), None, None),
    ("func_generic_type_array", _NDBuffer("float",1,False), "float[4]", 3),
    ("func_generic_length_array", _NDBuffer("float",1,False), None, None),
    ("func_generic_unsized_array", _NDBuffer("float",1,False), None, None),

    # standard structured buffer of known element type
    ("func_float_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), "StructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_float_rwstructuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), None, None),
    ("func_float_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=True), "StructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_float_rwstructuredbuffer", _Buffer(element_count=16, struct_size=4, rw=True), "RWStructuredBuffer<float,DefaultDataLayout>", 1),

    # As we don't do any type, size or element checking for basic buffers, any other types should also work
    ("func_half_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), "StructuredBuffer<half,DefaultDataLayout>", 1),
    ("func_int_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), "StructuredBuffer<int,DefaultDataLayout>", 1),

    # NDBuffer and Tensor can check types match
    ("func_float_structuredbuffer", _NDBuffer("float", 2, False), "StructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_float_rwstructuredbuffer", _NDBuffer("float", 2, False), None, None),
    ("func_float_rwstructuredbuffer", _NDBuffer("float", 2, True), "RWStructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_half_structuredbuffer", _NDBuffer("half", 2, False), "StructuredBuffer<half,DefaultDataLayout>", 1),
    ("func_int_structuredbuffer", _NDBuffer("int", 2, False), "StructuredBuffer<int,DefaultDataLayout>", 1),
    ("func_float_structuredbuffer", _Tensor("float", 2, False), "StructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_float_rwstructuredbuffer", _Tensor("float", 2, False), None, None),
    ("func_float_rwstructuredbuffer", _Tensor("float", 2, True), "RWStructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_half_structuredbuffer", _Tensor("half", 2, False),"StructuredBuffer<half,DefaultDataLayout>", 1),

    # Generic structured buffer can't be resolved with pure buffer (as it is typeless), but can
    # be resolved with NDBuffer / Tensor which have known types
    ("func_generic_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), None, None),
    ("func_generic_structuredbuffer", _NDBuffer("float", 2, False), "StructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_generic_structuredbuffer", _Tensor("float", 2, False), "StructuredBuffer<float,DefaultDataLayout>", 1),

    # ByteAddressBuffer / RWByteAddressBuffer for buffers of any type
    ("func_bytebuffer", _Buffer(element_count=16, struct_size=4, rw=False), "ByteAddressBuffer", 1),
    ("func_rwbytebuffer", _Buffer(element_count=16, struct_size=4, rw=False), None, None),
    ("func_rwbytebuffer", _Buffer(element_count=16, struct_size=4, rw=True), "RWByteAddressBuffer", 1),
    ("func_bytebuffer", _NDBuffer("float", 2, False), "ByteAddressBuffer", 1),
    ("func_rwbytebuffer", _NDBuffer("float", 2, False), None, None),
    ("func_rwbytebuffer", _NDBuffer("float", 2, True), "RWByteAddressBuffer", 1),
    ("func_bytebuffer", _Tensor("float", 2, False), "ByteAddressBuffer", 1),
    ("func_rwbytebuffer", _Tensor("float", 2, False), None, None),
    ("func_rwbytebuffer", _Tensor("float", 2, True), "RWByteAddressBuffer", 1),

    # Buffers as pointers
    ("func_float_ptr", _Buffer(element_count=16, struct_size=4, rw=False), "Ptr<float>", 1),
    ("func_float_ptr", _Buffer(element_count=16, struct_size=4, rw=True), "Ptr<float>", 1),
    ("func_float_ptr", _NDBuffer("float", 2, False), "Ptr<float>", 1),
    ("func_float_ptr", _NDBuffer("float", 2, True), "Ptr<float>", 1),
    ("func_float_ptr", _Tensor("float", 2, False), "Ptr<float>", 1),
    ("func_float_ptr", _Tensor("float", 2, True), "Ptr<float>", 1),
    ("func_float_cptr", _Buffer(element_count=16, struct_size=4, rw=False), "Ptr<float>", 1),
    ("func_float_cptr", _Buffer(element_count=16, struct_size=4, rw=True), "Ptr<float>", 1),
    ("func_float_cptr", _NDBuffer("float", 2, False), "Ptr<float>", 1),
    ("func_float_cptr", _NDBuffer("float", 2, True), "Ptr<float>", 1),
    ("func_float_cptr", _Tensor("float", 2, False), "Ptr<float>", 1),
    ("func_float_cptr", _Tensor("float", 2, True), "Ptr<float>", 1),

    # Int or buffer of pointers/uints
    ("func_float_ptr", 0, "Ptr<float>", 1),
    ("func_float_ptr", _NDBuffer("Ptr<float>", 2, False), "Ptr<float>", 1),
    ("func_float_ptr", _NDBuffer("uint64_t", 2, False), "Ptr<float>", 1),

    # Generic pointers can be resolved with typed NDBuffer/Tensor
    ("func_generic_ptr", _NDBuffer("float", 2, False), "Ptr<float>", 1),
    ("func_generic_ptr", _NDBuffer("float", 2, True), "Ptr<float>", 1),
    ("func_generic_ptr", _Tensor("float", 2, False), "Ptr<float>", 1),
    ("func_generic_ptr", _Tensor("float", 2, True), "Ptr<float>", 1),

]

# fmt: on


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name, arg_value, expected_type_name, expected_dim",
    TESTS,
    ids=[f"{fn}_{av}" for fn, av, etn, ed in TESTS],
)
def test_type_resolution_simple(
    device_type: spy.DeviceType,
    func_name: str,
    arg_value: Any,
    expected_type_name: Optional[str],
    expected_dim: Optional[int],
):
    device = helpers.get_device(type=device_type)
    module = spy.Module.load_from_file(device, "type_resolution.slang")

    if callable(arg_value):
        arg = arg_value(module)
    else:
        arg = arg_value

    try:
        actual_resolution = build_and_resolve(module, func_name, spyn.CallMode.prim, arg)
    except Exception as e:
        if expected_type_name is None:
            return
        else:
            raise e

    assert expected_type_name is not None, "Expected resolution to fail but it succeeded"
    expected_type = module.layout.find_type_by_name(expected_type_name)
    assert expected_type is not None, f"Expected type {expected_type_name} not found"
    check(actual_resolution, expected_type)
