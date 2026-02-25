# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, List, Optional, Tuple
import pytest

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
    unpacked_args, _ = spyn.unpack_args(*args)
    unpacked_kwargs, _ = spyn.unpack_kwargs(**kwargs)

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
    if len(resolutions) != 1:
        raise AssertionError(
            f"Expected one resolution, got {len(resolutions)}, diagnostics:\n" + str(diagnostics)
        )
    return resolutions[0]


def get_type(module: spy.Module, name: str):
    """Get a type from a module by name."""
    t = module.layout.find_type_by_name(name)
    assert t is not None, f"Type {name} not found"
    return t


@pytest.mark.skip("This is only a test for experimental vecotrization system")
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
def test_tensor_none_vectorize(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "int" if not generic else "T"
    sig_params = ", ".join([f"Tensor<{sig_param_type},1> p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.Tensor.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "Tensor<int,1>"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
def test_tensor_structured_buffer(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "int" if not generic else "T"
    sig_params = ", ".join([f"StructuredBuffer<{sig_param_type}> p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.Tensor.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "StructuredBuffer<int>"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False])  # Fully generic not supported
def test_tensor_vectorize(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "int" if not generic else "T"
    sig_params = ", ".join([f"{sig_param_type} p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.Tensor.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "int"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
def test_tensor_vectorize_vectortype(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "vector<int,1>" if not generic else "vector<T,1>"
    sig_params = ", ".join([f"{sig_param_type} p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.Tensor.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "vector<int,1>"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
def test_tensor_vectorize_arraytype(device_type: spy.DeviceType, param_count: int, generic: bool):
    device = helpers.get_device(type=device_type)

    sig_generic = "" if not generic else "<T>"
    sig_param_type = "int[1]" if not generic else "T[1]"
    sig_params = ", ".join([f"{sig_param_type} p{i}" for i in range(param_count)])
    code = f"""
import slangpy;
void test_func{sig_generic}({sig_params}) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.Tensor.empty(device, (10,), dtype=get_type(module, "int")),) * param_count
    param_types = (get_type(module, "int[1]"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [False, True])
def test_tensorarray_vectorize_arraytype(
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
        spy.Tensor.empty(device, (10,), dtype=get_type(module, "int[1]")),
    ) * param_count
    param_types = (get_type(module, "int[1]"),) * param_count
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("param_count", [1, 3])
@pytest.mark.parametrize("generic", [True])  # Only generic makes sense here
def test_tensorarray_vectorize_arraytype2(
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
        spy.Tensor.empty(device, (10,), dtype=get_type(module, "int[1]")),
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

    param_values = (spy.Tensor.empty(device, (10,), dtype=get_type(module, "int")),)
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
def test_tensorarray_generic_struct(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    code = f"""
import slangpy;

struct Foo<T> {{ T value; }}

void test_func<T>(Foo<T> p0) {{}}
"""
    module = helpers.create_module(device, code)

    param_values = (spy.Tensor.empty(device, (10,), dtype=get_type(module, "Foo<int>")),)
    param_types = (get_type(module, "Foo<int>"),)
    actual_resolution = build_and_resolve(module, "test_func", spyn.CallMode.prim, *param_values)
    check(actual_resolution, *param_types)


def calc_usage(rw: bool) -> spy.BufferUsage:
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    return usage


def calc_tex_usage(rw: bool) -> spy.TextureUsage:
    usage = spy.TextureUsage.shader_resource
    if rw:
        usage |= spy.TextureUsage.unordered_access
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


class _Tensor:
    def __init__(
        self, base_type: str, dim: int, rw: bool, gradin: bool = False, gradout: bool = False
    ):
        super().__init__()
        self.base_type = base_type
        self.dim = dim
        self.rw = rw
        self.gradin = gradin
        self.gradout = gradout

    def __repr__(self) -> str:
        txt = f"{'RW' if self.rw else ''}Tensor<{self.base_type},{self.dim}"
        if self.gradin:
            txt += ", gradin"
        if self.gradout:
            txt += ", gradout"
        txt += ">"
        return txt

    def __call__(self, module: spy.Module) -> spy.Tensor:
        t = spy.Tensor.empty(
            module.device,
            (3,) * self.dim,
            dtype=module.layout.find_type_by_name(self.base_type),
            usage=calc_usage(self.rw),
        )
        if self.gradin or self.gradout:
            t = t.with_grads(
                grad_in=spy.Tensor.empty_like(t) if self.gradin else None,
                grad_out=spy.Tensor.empty_like(t) if self.gradout else None,
            )
        return t


class _Texture:
    def __init__(
        self,
        type: spy.TextureType = spy.TextureType.texture_2d,
        format: spy.Format = spy.Format.undefined,
        rw: bool = False,
    ):
        super().__init__()
        self.texture_type = type
        self.format = format
        self.width = 16
        self.height = (
            16
            if type
            in (
                spy.TextureType.texture_2d,
                spy.TextureType.texture_2d_array,
                spy.TextureType.texture_cube,
                spy.TextureType.texture_cube_array,
            )
            else 1
        )
        self.depth = 16 if type == spy.TextureType.texture_3d else 1
        self.array_length = (
            4
            if type
            in (
                spy.TextureType.texture_1d_array,
                spy.TextureType.texture_2d_array,
                spy.TextureType.texture_cube_array,
            )
            else 1
        )
        self.rw = rw

    def __repr__(self) -> str:
        name_map = {
            spy.TextureType.texture_1d: "Texture1D",
            spy.TextureType.texture_2d: "Texture2D",
            spy.TextureType.texture_3d: "Texture3D",
            spy.TextureType.texture_cube: "TextureCube",
            spy.TextureType.texture_1d_array: "Texture1DArray",
            spy.TextureType.texture_2d_array: "Texture2DArray",
            spy.TextureType.texture_cube_array: "TextureCubeArray",
        }
        return (
            f"{'RW' if self.rw else ''}{name_map.get(self.texture_type, 'Texture')}<{self.format}>"
        )

    def __call__(self, module: spy.Module) -> spy.Texture:

        return module.device.create_texture(
            type=self.texture_type,
            format=self.format,
            width=self.width,
            height=self.height,
            depth=self.depth,
            array_length=self.array_length,
            usage=calc_tex_usage(self.rw),
        )


# TorchTensor helper for testing TensorView type resolution (CUDA-only)
try:
    import torch

    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()

    # Import torch integration to register torch.Tensor handler
    if _HAS_CUDA:
        import slangpy.torchintegration.torchtensormarshall  # noqa: F401
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False


class _TorchTensor:
    _dtype_map = {
        "float": "float32",
        "int": "int32",
        "half": "float16",
        "int8_t": "int8",
        "int16_t": "int16",
        "int64_t": "int64",
        "uint8_t": "uint8",
    }

    def __init__(self, base_type: str, dim: int):
        super().__init__()
        self.base_type = base_type
        self.dim = dim

    def __repr__(self) -> str:
        return f"TorchTensor<{self.base_type},{self.dim}>"

    def __call__(self, module: spy.Module):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not installed")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # Map Slang type name to torch dtype
        torch_dtype_name = self._dtype_map.get(self.base_type)
        if torch_dtype_name is None:
            raise ValueError(f"Unknown torch dtype for Slang type {self.base_type}")
        torch_dtype = getattr(torch, torch_dtype_name)

        return torch.zeros((3,) * self.dim, dtype=torch_dtype, device="cuda")


class _TorchTensorDiffPair:
    _dtype_map = {
        "float": "float32",
        "int": "int32",
        "half": "float16",
    }

    def __init__(self, base_type: str, dim: int):
        super().__init__()
        self.base_type = base_type
        self.dim = dim

    def __repr__(self) -> str:
        return f"TorchTensorDiffPair<{self.base_type},{self.dim}>"

    def __call__(self, module: spy.Module):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not installed")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        from slangpy.core.native import NativeTorchTensorDiffPair

        torch_dtype_name = self._dtype_map.get(self.base_type)
        if torch_dtype_name is None:
            raise ValueError(f"Unknown torch dtype for Slang type {self.base_type}")
        torch_dtype = getattr(torch, torch_dtype_name)

        primal = torch.zeros((3,) * self.dim, dtype=torch_dtype, device="cuda")
        return NativeTorchTensorDiffPair(primal, None)


# fmt: off

# List of simple tests, where each is a function, input argument value (or factor func + args), and expected resolved type / dimensionality.
# Functions are in type_resolution.slang
TESTS = [
    # Standard scalar float tests
    ("func_float", 0, "float", 0),
    ("func_float", 1.5, "float", 0),
    ("func_float", spy.float1(2.5), "float", 1),
    ("func_float", spy.float3(3.5, 4.5, 5.5), "float", 3),
    ("func_float", _Tensor("float", 1, False), "float", 1),
    ("func_float", _Tensor("float", 2, True), "float", 2),
    ("func_float",[1.5], "float", 1),
    ("func_float",[1], "float", 1),

    # These should fail as we don't implicit cast AND vectorize - its one or the other
    ("func_float", spy.int3(0,0,0), None, None),
    ("func_float", _Tensor("int", 1, False), None, None),

    # Standard scalar int tests
    ("func_int", 42, "int", 0),
    ("func_int", 1.5, "int", 0), # Passes, as invalid fractional value has to be detected at runtime
    ("func_int", spy.int1(2), "int", 1),

    # Python ranges to ints but not floats
    ("func_int", range(3), "int", 3),
    ("func_float", range(3), None, None),
    ("func_generic", range(3), "int", 3),

    # None default numeric types should be picked up correctly
    ("func_half", 1.0, "half", 0),
    ("func_half", 2, "half", 0),
    ("func_int8", 255, "int8_t", 0),
    ("func_int64", 2**64 - 1, "int64_t", 0),

    # Buffer/tensor -> function
    ("func_half", _Tensor("half", 1, False), "half", 1),

    # Fully generic, only unambiguous for types that can't be vectorized
    ("func_generic", 3.5, "float", 0),
    ("func_generic", 42, "int", 0),
    ("func_generic", spy.float1(2.5), None, None),
    ("func_generic", [42], None, None),
    ("func_generic", _Tensor("float", 2, True), None, None),

    # Generic constrained to built in integer allows vectorization
    ("func_genericint", 42, "int", 0),
    ("func_genericint", spy.int3(1,2,3), "int", 3),

    # Same for floating point
    ("func_genericfloat", 42.5, "float", 0),
    ("func_genericfloat", spy.float3(1,2,3), "float", 3),
    ("func_genericfloat", _Tensor("float", 2, False), "float", 2),

    # Also ensure specialization fails when types don't match
    ("func_genericint", _Tensor("float", 2, False), None, None),

    # float3 tests
    ("func_float3", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_float3", [1.0, 2.0, 3.0], "vector<float,3>", 3),
    ("func_float3", _Tensor("float", 2, True), "vector<float,3>", 1),
    ("func_float3", _Tensor("float3", 1, True), "vector<float,3>", 1),
    ("func_float3", _Tensor("float4", 1, True), None, None),

    # Implicit scalar -> vector cast, as it is so useful, binds as the scalar type
    # and relies on slang to do the upcast.
    ("func_float3", 1.0, "float", 0),
    ("func_int3", 1.0, "int", 0),
    ("func_half3", 1.0, "half", 0),

    # slang arg defined as vector<float,3> to ensure identical resolution
    ("func_vector_float3", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_vector_float3", _Tensor("float", 2, True), "vector<float,3>", 1),

    # Other vector types
    ("func_int3", spy.int3(1, 2, 3), "vector<int,3>", 3),
    ("func_half3", _Tensor("half3", 1, True), "vector<half,3>", 1),

    # We should potentially allow these implicit casts as there is no vectorizing
    # happening, but for now we don't.
    ("func_half3", spy.float3(1.0, 2.0, 3.0), None, None),

    # generic typed vector tests
    ("func_vector3_generic", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_vector3_generic", _Tensor("float", 2, True), "vector<float,3>", 1),
    ("func_vector3_generic", _Tensor("float3", 1, True), "vector<float,3>", 1),
    ("func_vector3_generic", _Tensor("float4", 1, True), None, None),

    # generic typed+dim vector tests
    ("func_vectorN_generic", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_vectorN_generic", _Tensor("float", 2, True), None, None),
    ("func_vectorN_generic", _Tensor("float3", 1, True), "vector<float,3>", 1),
    ("func_vectorN_generic", _Tensor("float4", 1, True), "vector<float,4>", 1),

    # generic dim float tests (similar, but verify incorrect element type fails)
    ("func_floatN_generic", spy.float3(1.0, 2.0, 3.0), "vector<float,3>", 3),
    ("func_floatN_generic", [1.0, 2.0, 3.0], "vector<float,3>", 3),
    ("func_floatN_generic", _Tensor("float", 2, True), None, None),
    ("func_floatN_generic", _Tensor("float3", 1, True), "vector<float,3>", 1),
    ("func_floatN_generic", _Tensor("float4", 1, True), "vector<float,4>", 1),

    # Load vectors from matrices
    ("func_float3", spy.float3x3.identity(), "vector<float,3>", 2),
    ("func_vector_float3", spy.float3x3.identity(), "vector<float,3>", 2),
    ("func_vector3_generic", spy.float3x3.identity(), "vector<float,3>", 2),
    ("func_vectorN_generic", spy.float3x3.identity(), "vector<float,3>", 2),
    ("func_floatN_generic", spy.float3x3.identity(), "vector<float,3>", 2),

    # Basic matrix types
    ("func_float3x3", spy.float3x3.identity(), "matrix<float,3,3>", 2),
    ("func_float3x3", _Tensor("float", 2, True), "matrix<float,3,3>", 1),
    ("func_float3x3", _Tensor("float3x3", 1, True), "matrix<float,3,3>", 1),
    ("func_float3x3", _Tensor("float4x4", 1, True), None, None),
    ("func_float3x4", spy.float3x4.identity(), "matrix<float,3,4>", 2),
    ("func_matrix_float3x3", spy.float3x3.identity(), "matrix<float,3,3>", 2),
    ("func_matrix_generic", spy.float3x3.identity(), "matrix<float,3,3>", 2),
    ("func_matrix_generic33", spy.float3x3.identity(), "matrix<float,3,3>", 2),
    ("func_matrix_floatRC_generic", spy.float3x3.identity(), "matrix<float,3,3>", 2),
    ("func_matrix_floatR3_generic", spy.float3x3.identity(), "matrix<float,3,3>", 2),
    ("func_matrix_float3C_generic", spy.float3x3.identity(), "matrix<float,3,3>", 2),

    # Loading matrices from buffers/tensors of scalars require known dimensions but can resolve type
    ("func_matrix_generic", _Tensor("float", 2, False), None, None),
    ("func_matrix_generic33", _Tensor("float", 2, False), "matrix<float,3,3>", 2),
    ("func_matrix_floatRC_generic", _Tensor("float", 2, False), None, None),
    ("func_matrix_floatR3_generic", _Tensor("float", 2, False), None, None),
    ("func_matrix_float3C_generic", _Tensor("float", 2, False), None, None),

    # Loading generic matrices from buffers/tensors of matrices
    ("func_matrix_generic", _Tensor("float3x3", 1, True), "matrix<float,3,3>", 1),
    ("func_matrix_generic33", _Tensor("float3x3", 1, True), "matrix<float,3,3>", 1),
    ("func_matrix_floatRC_generic", _Tensor("float3x4", 1, True), "matrix<float,3,4>", 1),
    ("func_matrix_floatR3_generic", _Tensor("float3x3", 1, True), "matrix<float,3,3>", 1),
    ("func_matrix_float3C_generic", _Tensor("float4x3", 1, True), None, None),
    ("func_matrix_float3C_generic", _Tensor("float3x4", 1, True), "matrix<float,3,4>", 1),

    # Basic float arrays
    ("func_float_array", [1,2,3,4], "float[4]", 3),
    ("func_float_array", _Tensor("float[4]",1,False), "float[4]", 3),
    ("func_float_array2", [1,2,3,4,5,6,7,8], "float[8]", 3),
    ("func_float_array2", _Tensor("float[8]",1,False), "float[8]", 3),

    # Incorrect sizes/types
    ("func_float_array2", [1,2,3,4], None, None),

    # Unsized / generic arrays
    ("func_float_unsized_array", [1,2,3,4], "float[4]", 3),
    ("func_generic_array", [1,2,3,4], "int[4]", 3),
    ("func_generic_array", _Tensor("float[4]",1,False), "float[4]", 3),
    ("func_generic_type_array", [1.5,2.5,3.5,4.6], "float[4]", 3),
    ("func_generic_length_array", [1.5,2.5,3.5,4.6], "float[4]", 3),
    ("func_generic_length_array", _Tensor("float[4]",1,False), "float[4]", 3),
    ("func_generic_unsized_array", [1.5,2.5,3.5,4.6], "float[4]", 3),
    ("func_generic_unsized_array", _Tensor("float[4]",1,False), "float[4]", 3),

    # These are ambiguous, as the function has T[4], so could resolve with T==float
    # or T==float[4].
    ("func_generic_type_array", _Tensor("float[4]",1,False), None, None),

    # Although dimension doesn't match (the function is T[4]), the fact that the
    # array is completely generic means this can resolve with T==float[8]
    ("func_generic_type_array", _Tensor("float[8]",1,False), "float[8][4]", 1),

    # The constrained generic type (to __BuiltinFloatingPointType) means that
    # T==float is the only valid resolution, so it is no longer ambiguous
    ("func_generic_constrained_type_array", _Tensor("float[4]",1,False), "float[4]", 1),

    # Generic array resolutions that should fail due to wrong size/type
    ("func_generic_length_array", _Tensor("int[4]",1,False), None, None),

    # Loading from container of floats to array of floats. As size can not
    # be determined at compile time, none finite length arrays can't be resolved.
    ("func_float_array", _Tensor("float",1,True), "float[4]", 3),
    ("func_float_array2", _Tensor("float",1,True), "float[8]", 3),
    ("func_float_unsized_array", _Tensor("float",1,False), None, None),
    ("func_generic_array", _Tensor("float",1,False), None, None),
    ("func_generic_type_array", _Tensor("float",1,False), "float[4]", 3),
    ("func_generic_length_array", _Tensor("float",1,False), None, None),
    ("func_generic_unsized_array", _Tensor("float",1,False), None, None),

    # 2D float arrays
    ("func_float_array2d", _Tensor("float[8][5]",1,False), "float[8][5]", 3),
    ("func_float_array2d_full", _Tensor("float[8][5]",1,False), "float[8][5]", 3),
    ("func_float_array2d", _Tensor("float",2,False), "float[8][5]", 3),
    ("func_float_array2d_full", _Tensor("float",2,False), "float[8][5]", 3),

    # Tensor of 1D arrays that can map to the 2D array
    ("func_float_array2d", _Tensor("float[8]",1,False), "float[8][5]", 3),

    # Failure cases due to incorrect sizes/types
    ("func_float_array2d", _Tensor("float[5][8]",1,False), None, None),
    ("func_float_array2d", _Tensor("float[5]",1,False), None, None),

    # Generic types array here can't work, as could end up with float[8][5]
    # or float[5][8][5] or float[5][8][5][8]!!!
    ("func_generic_type_array2d", _Tensor("float[8][5]",1,True), None, None),

    # For the same reason, these are ambiguous
    ("func_generic_type_array2d", _Tensor("float[8]",1,True), None, None),

    # These can succeed, as the function is constrained to require T is floating point
    ("func_generic_type_constrained_array2d", _Tensor("float[8][5]",1,True), "float[8][5]", 1),

    # These can succeed, as the different dimensions means only T==float[3][4][8][5] works!
    ("func_generic_type_array2d", _Tensor("float[3][4]",1,True), "float[3][4][8][5]", 3),

    # As there is only 1 solution to matching up types, these are not ambiguous
    ("func_generic_type_array2d", _Tensor("float",1,True), "float[8][5]", 1),
    ("func_generic_type_array2d", _Tensor("float[3]",1,True), "float[3][8][5]", 1),
    ("func_generic_type_array2d", _Tensor("float[3]",1,True), "float[3][8][5]", 1),

    # Resolve dimensions of generically sized arrays
    ("func_generic_size_array2d_R", _Tensor("float[8][5]",1,True), "float[8][5]", 1),
    ("func_generic_size_array2d_C", _Tensor("float[8][5]",1,True), "float[8][5]", 1),
    ("func_generic_size_array2d_RC", _Tensor("float[8][5]",1,True), "float[8][5]", 1),

    # With just R undefined, the system has a known C, so it can take the R from
    # the element type and the C from the function signature. With C or RC undefined
    # there is no way to resolve C, so those fail.
    ("func_generic_size_array2d_R", _Tensor("float[8]",1,True), "float[8][5]", 1),
    ("func_generic_size_array2d_C", _Tensor("float[8]",1,True), None, None),
    ("func_generic_size_array2d_RC", _Tensor("float[8]",1,True), None, None),

    # standard structured buffer of known element type
    ("func_float_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), "StructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_float_rwstructuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), None, None),
    ("func_float_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=True), "StructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_float_rwstructuredbuffer", _Buffer(element_count=16, struct_size=4, rw=True), "RWStructuredBuffer<float,DefaultDataLayout>", 1),

    # As we don't do any type, size or element checking for basic buffers, any other types should also work
    ("func_half_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), "StructuredBuffer<half,DefaultDataLayout>", 1),
    ("func_int_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), "StructuredBuffer<int,DefaultDataLayout>", 1),

    # Tensor can check types match
    ("func_int_structuredbuffer", _Tensor("int", 2, False), "StructuredBuffer<int,DefaultDataLayout>", 1),
    ("func_float_structuredbuffer", _Tensor("float", 2, False), "StructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_float_rwstructuredbuffer", _Tensor("float", 2, False), None, None),
    ("func_float_rwstructuredbuffer", _Tensor("float", 2, True), "RWStructuredBuffer<float,DefaultDataLayout>", 1),
    ("func_half_structuredbuffer", _Tensor("half", 2, False),"StructuredBuffer<half,DefaultDataLayout>", 1),

    # Generic structured buffer can't be resolved with pure buffer (as it is typeless), but can
    # be resolved with Tensor which have known types
    ("func_generic_structuredbuffer", _Buffer(element_count=16, struct_size=4, rw=False), None, None),
    ("func_generic_structuredbuffer", _Tensor("float", 2, False), "StructuredBuffer<float,DefaultDataLayout>", 1),

    # ByteAddressBuffer / RWByteAddressBuffer for buffers of any type
    ("func_bytebuffer", _Buffer(element_count=16, struct_size=4, rw=False), "ByteAddressBuffer", 1),
    ("func_rwbytebuffer", _Buffer(element_count=16, struct_size=4, rw=False), None, None),
    ("func_rwbytebuffer", _Buffer(element_count=16, struct_size=4, rw=True), "RWByteAddressBuffer", 1),
    ("func_bytebuffer", _Tensor("float", 2, False), "ByteAddressBuffer", 1),
    ("func_rwbytebuffer", _Tensor("float", 2, False), None, None),
    ("func_rwbytebuffer", _Tensor("float", 2, True), "RWByteAddressBuffer", 1),

    # Buffers as pointers
    ("func_float_ptr", _Buffer(element_count=16, struct_size=4, rw=False), "Ptr<float>", 1),
    ("func_float_ptr", _Buffer(element_count=16, struct_size=4, rw=True), "Ptr<float>", 1),
    ("func_float_ptr", _Tensor("float", 2, False), "Ptr<float>", 1),
    ("func_float_ptr", _Tensor("float", 2, True), "Ptr<float>", 1),
    ("func_float_cptr", _Buffer(element_count=16, struct_size=4, rw=False), "Ptr<float>", 1),
    ("func_float_cptr", _Buffer(element_count=16, struct_size=4, rw=True), "Ptr<float>", 1),
    ("func_float_cptr", _Tensor("float", 2, False), "Ptr<float>", 1),
    ("func_float_cptr", _Tensor("float", 2, True), "Ptr<float>", 1),

    # Int or buffer of pointers/uints
    ("func_float_ptr", 0, "Ptr<float>", 1),
    ("func_float_ptr", _Tensor("Ptr<float>", 2, False), "Ptr<float>", 1),
    ("func_float_ptr", _Tensor("uint64_t", 2, False), "Ptr<float>", 1),

    # Generic pointers can be resolved with typed Tensor
    ("func_generic_ptr", _Tensor("float", 2, False), "Ptr<float>", 1),
    ("func_generic_ptr", _Tensor("float", 2, True), "Ptr<float>", 1),

    # Basic passing of Tensor to the internal Tensor types
    ("func_iTensor", _Tensor("float", 2, False), "Tensor<float,2>", 2),
    ("func_iTensor", _Tensor("float", 2, True), "Tensor<float,2>", 2),
    ("func_iWTensor", _Tensor("float", 2, False), None, None),
    ("func_iWTensor", _Tensor("float", 2, True), "WTensor<float,2>", 2),
    ("func_irwtensor", _Tensor("float", 2, False), None, None),
    ("func_irwtensor", _Tensor("float", 2, True), "RWTensor<float,2>", 2),

    ("func_Tensor", _Tensor("float", 2, False), "Tensor<float,2>", 2),
    ("func_Tensor", _Tensor("float", 2, True), "Tensor<float,2>", 2),
    ("func_WTensor", _Tensor("float", 2, False), None, None),
    ("func_WTensor", _Tensor("float", 2, True), "WTensor<float,2>", 2),
    ("func_rwtensor", _Tensor("float", 2, False), None, None),
    ("func_rwtensor", _Tensor("float", 2, True), "RWTensor<float,2>", 2),

    ("func_atomictensor", _Tensor("float", 2, False), None, None),
    ("func_atomictensor", _Tensor("float", 2, True), "AtomicTensor<float,2>", 2),

    # Diff tensor interfaces should work without grads and map to primal tensors,
    # because these tests are run for the primal pass.
    ("func_iDiffTensor", _Tensor("float", 2, False), "PrimalTensor<float,2>", 2),
    ("func_iDiffTensor", _Tensor("float", 2, True), "PrimalTensor<float,2>", 2),
    ("func_iWDiffTensor", _Tensor("float", 2, False), None, None),
    ("func_iWDiffTensor", _Tensor("float", 2, True), "WPrimalTensor<float,2>", 2),
    ("func_irwdifftensor", _Tensor("float", 2, False), None, None),
    ("func_irwdifftensor", _Tensor("float", 2, True), "RWPrimalTensor<float,2>", 2),
    ("func_iDiffTensor", _Tensor("float", 2, True, True, True), "PrimalTensor<float,2>", 2),
    ("func_iWDiffTensor", _Tensor("float", 2, True, True, True), "WPrimalTensor<float,2>", 2),
    ("func_irwdifftensor", _Tensor("float", 2, True, True, True), "RWPrimalTensor<float,2>", 2),

    # Primal tensors are the same, and should work with / without grads
    ("func_PrimalTensor", _Tensor("float", 2, False), "PrimalTensor<float,2>", 2),
    ("func_PrimalTensor", _Tensor("float", 2, True), "PrimalTensor<float,2>", 2),
    ("func_WPrimalTensor", _Tensor("float", 2, False), None, None),
    ("func_WPrimalTensor", _Tensor("float", 2, True), "WPrimalTensor<float,2>", 2),
    ("func_rwprimaltensor", _Tensor("float", 2, False), None, None),
    ("func_rwprimaltensor", _Tensor("float", 2, True), "RWPrimalTensor<float,2>", 2),
    ("func_PrimalTensor", _Tensor("float", 2, True, True, True), "PrimalTensor<float,2>", 2),
    ("func_WPrimalTensor", _Tensor("float", 2, True, True, True), "WPrimalTensor<float,2>", 2),
    ("func_rwprimaltensor", _Tensor("float", 2, True, True, True), "RWPrimalTensor<float,2>", 2),

    # Any differentiable tensors require differentiable types
    ("func_iDiffTensor", _Tensor("int", 2, False), None, None),
    ("func_iDiffTensor", _Tensor("int", 2, True), None, None),
    ("func_iWDiffTensor", _Tensor("int", 2, False), None, None),
    ("func_iWDiffTensor", _Tensor("int", 2, True), None, None),
    ("func_irwdifftensor", _Tensor("int", 2, False), None, None),
    ("func_irwdifftensor", _Tensor("int", 2, True), None, None),

    # Diff tensors require grads so these should all fail
    ("func_DiffTensor", _Tensor("float", 2, False), None, None),
    ("func_DiffTensor", _Tensor("float", 2, True), None, None),
    ("func_WDiffTensor", _Tensor("float", 2, False), None, None),
    ("func_WDiffTensor", _Tensor("float", 2, True), None, None),
    ("func_rwdifftensor", _Tensor("float", 2, False), None, None),
    ("func_rwdifftensor", _Tensor("float", 2, True), None, None),

    # Diff tensors with the wrong grads should also fail!
    ("func_DiffTensor", _Tensor("float", 2, True, True, False), None, None),
    ("func_WDiffTensor", _Tensor("float", 2, True, False, True), None, None),
    ("func_rwdifftensor", _Tensor("float", 2, True, False, True), None, None),
    ("func_rwdifftensor", _Tensor("float", 2, True, True, False), None, None),

    # Diff tensors with the correct grads should also pass!
    ("func_DiffTensor", _Tensor("float", 2, True, False, True), "DiffTensor<float,2>", 2),
    ("func_WDiffTensor", _Tensor("float", 2, True, True, False), "WDiffTensor<float,2>", 2),
    ("func_rwdifftensor", _Tensor("float", 2, True, True, True), "RWDiffTensor<float,2>", 2),

    # Generic typed tensors
    ("func_generic_element_tensor", _Tensor("float", 2, True), "Tensor<float,2>", 2),
    ("func_generic_dims_tensor", _Tensor("float", 2, True), "Tensor<float,2>", 2),
    ("func_generic_tensor", _Tensor("float", 2, True), "Tensor<float,2>", 2),

    # Float texture loading into floats/vectors
    ("func_float", _Texture(spy.TextureType.texture_1d, spy.Format.r32_float, False), "float", 2),
    ("func_genericfloat", _Texture(spy.TextureType.texture_1d, spy.Format.r32_float, False), "float", 2),
    ("func_float3", _Texture(spy.TextureType.texture_1d, spy.Format.rgb32_float, False), "vector<float,3>", 2),
    ("func_floatN_generic", _Texture(spy.TextureType.texture_1d, spy.Format.r32_float, False), "vector<float,1>", 2),
    ("func_floatN_generic", _Texture(spy.TextureType.texture_1d, spy.Format.rg32_float, False), "vector<float,2>", 2),
    ("func_floatN_generic", _Texture(spy.TextureType.texture_1d, spy.Format.rgb32_float, False), "vector<float,3>", 2),
    ("func_vector3_generic", _Texture(spy.TextureType.texture_1d, spy.Format.rgb32_float, False), "vector<float,3>", 2),
    ("func_floatN_generic", _Texture(spy.TextureType.texture_1d, spy.Format.rgba32_float, False), "vector<float,4>", 2),
    ("func_vectorN_generic", _Texture(spy.TextureType.texture_1d, spy.Format.rgba32_float, False), "vector<float,4>", 2),

    # Int texture loading into ints/vectors
    ("func_int", _Texture(spy.TextureType.texture_1d, spy.Format.r32_sint, False), "int", 2),
    ("func_genericint", _Texture(spy.TextureType.texture_1d, spy.Format.r32_sint, False), "int", 2),
    ("func_int3", _Texture(spy.TextureType.texture_1d, spy.Format.rgb32_sint, False), "vector<int,3>", 2),
    ("func_vector3_generic", _Texture(spy.TextureType.texture_1d, spy.Format.rgb32_sint, False), "vector<int,3>", 2),
    ("func_vectorN_generic", _Texture(spy.TextureType.texture_1d, spy.Format.rgba32_sint, False), "vector<int,4>", 2),

    # Various texture types
    ("func_texture1d", _Texture(spy.TextureType.texture_1d, spy.Format.rgba32_float, False), "Texture1D<float4>", 0),
    ("func_rwtexture1d", _Texture(spy.TextureType.texture_1d, spy.Format.rgba32_float, False), None, None),
    ("func_rwtexture1d", _Texture(spy.TextureType.texture_1d, spy.Format.rgba32_float, True), "RWTexture1D<float4>", 0),
    ("func_texture2d", _Texture(spy.TextureType.texture_2d, spy.Format.rgba32_float, False), "Texture2D<float4>", 0),
    ("func_rwtexture2d", _Texture(spy.TextureType.texture_2d, spy.Format.rgba32_float, False), None, None),
    ("func_rwtexture2d", _Texture(spy.TextureType.texture_2d, spy.Format.rgba32_float, True), "RWTexture2D<float4>", 0),
    ("func_texture3d", _Texture(spy.TextureType.texture_3d, spy.Format.rgba32_float, False), "Texture3D<float4>", 0),
    ("func_rwtexture3d", _Texture(spy.TextureType.texture_3d, spy.Format.rgba32_float, False), None, None),
    ("func_rwtexture3d", _Texture(spy.TextureType.texture_3d, spy.Format.rgba32_float, True), "RWTexture3D<float4>", 0),
    ("func_texture1darray", _Texture(spy.TextureType.texture_1d_array, spy.Format.rgba32_float, False), "Texture1DArray<float4>", 0),
    ("func_rwtexture1darray", _Texture(spy.TextureType.texture_1d_array, spy.Format.rgba32_float, False), None, None),
    ("func_rwtexture1darray", _Texture(spy.TextureType.texture_1d_array, spy.Format.rgba32_float, True), "RWTexture1DArray<float4>", 0),
    ("func_texture2darray", _Texture(spy.TextureType.texture_2d_array, spy.Format.rgba32_float, False), "Texture2DArray<float4>", 0),
    ("func_rwtexture2darray", _Texture(spy.TextureType.texture_2d_array, spy.Format.rgba32_float, False), None, None),
    ("func_rwtexture2darray", _Texture(spy.TextureType.texture_2d_array, spy.Format.rgba32_float, True), "RWTexture2DArray<float4>", 0),

    # Generic texture types
    ("func_texture2d_generic", _Texture(spy.TextureType.texture_1d, spy.Format.rgba32_float, False), None, None),
    ("func_texture2d_generic", _Texture(spy.TextureType.texture_2d, spy.Format.r32_float, False), "Texture2D<float1>", 0),
    ("func_texture2d_generic", _Texture(spy.TextureType.texture_2d, spy.Format.rgba32_float, False), "Texture2D<float4>", 0),
    ("func_rwtexture2d_generic", _Texture(spy.TextureType.texture_2d, spy.Format.rgba32_float, False), None, None),
    ("func_rwtexture2d_generic", _Texture(spy.TextureType.texture_2d, spy.Format.rgba32_float, True), "RWTexture2D<float4>", 0),

    # Basic struct tests
    ("func_struct", {}, "Foo", 0),
    ("func_struct", {"_type": "Foo"}, "Foo", 0),
    ("func_struct", {"_type": "NotFoo"}, None, None),
    ("func_generic_struct", {}, None, None),
    ("func_generic_struct", {"_type": "GenericFoo<int>"}, "GenericFoo<int>", 0),
    ("func_generic_struct", {"_type": "GenericFoo<float>"}, "GenericFoo<float>", 0),
    ("func_generic_struct_int", {}, "GenericFoo<int>", 0),
    ("func_generic_struct_int", {"_type": "GenericFoo<int>"}, "GenericFoo<int>", 0),
    ("func_generic_struct_int", {"_type": "GenericFoo<float>"}, None, None),

    # Tensor of structs
    ("func_struct", _Tensor("Foo", 2, False), "Foo", 2),
    ("func_generic_struct", _Tensor("GenericFoo<int>", 2, False), "GenericFoo<int>", 2),
    ("func_generic_struct_int", _Tensor("GenericFoo<int>", 2, False), "GenericFoo<int>", 2),
    ("func_generic_struct_int", _Tensor("GenericFoo<float>", 2, False), None, None),

    # Array of structs
    ("func_struct", [{}], "Foo", 1),
    ("func_struct", [{"_type": "Foo"}], "Foo", 1),
    ("func_struct", [{"_type": "NotFoo"}], None, None),

    # Dictionary to vector
    ("func_float3", {"x":1.0, "y":2.0, "z":3.0}, "vector<float,3>", 0),
    ("func_vector3_generic", {"_type":"float3", "x":1.0, "y":2.0, "z":3.0}, "vector<float,3>", 2),
    ("func_floatN_generic", {"_type":"float3", "x":1.0, "y":2.0, "z":3.0}, "vector<float,3>", 2),
    ("func_vectorN_generic", {"_type":"float3", "x":1.0, "y":2.0, "z":3.0}, "vector<float,3>", 2),

    # Interfaces
    ("func_interface", {}, None, None),
    ("func_interface", {"_type": "Bar"}, "Bar", 0),
    ("func_interface", {"_type": "Foo"}, None, None),
    ("func_interface", [{}], None, None),
    ("func_interface", [{"_type": "Bar"}], "Bar", 0),
    ("func_interface", [{"_type": "Foo"}], None, None),
    ("func_interface", _Tensor("Bar", 2, True), "Bar", 2),

    # TensorView type resolution tests (CUDA-only, requires PyTorch)
    # torch.Tensor should resolve to TensorView<T> when binding to TensorView parameter
    ("func_tensorview_float", _TorchTensor("float", 1), "TensorView<float>", 1),
    ("func_tensorview_float", _TorchTensor("float", 2), "TensorView<float>", 2),
    ("func_tensorview_int", _TorchTensor("int", 1), "TensorView<int>", 1),
    ("func_tensorview_half", _TorchTensor("half", 2), "TensorView<half>", 2),

    # Type mismatch should fail
    ("func_tensorview_float", _TorchTensor("int", 1), None, None),
    ("func_tensorview_int", _TorchTensor("float", 1), None, None),

    # Generic TensorView<T> resolution
    ("func_tensorview_generic", _TorchTensor("float", 1), "TensorView<float>", 1),
    ("func_tensorview_generic", _TorchTensor("int", 2), "TensorView<int>", 2),

    # Normal slangpy Tensor should also resolve to TensorView<T>
    ("func_tensorview_float", _Tensor("float", 1, True), "TensorView<float>", 1),
    ("func_tensorview_float", _Tensor("float", 2, True), "TensorView<float>", 2),
    ("func_tensorview_float", _Tensor("float", 1, False), "TensorView<float>", 1),
    ("func_tensorview_int", _Tensor("int", 1, True), "TensorView<int>", 1),

    # Normal slangpy Tensor type mismatch should fail
    ("func_tensorview_float", _Tensor("int", 1, True), None, None),
    ("func_tensorview_int", _Tensor("float", 1, True), None, None),

    # Normal slangpy Tensor with generic TensorView<T>
    ("func_tensorview_generic", _Tensor("float", 1, True), "TensorView<float>", 1),
    ("func_tensorview_generic", _Tensor("int", 2, True), "TensorView<int>", 2),

    # TensorView<float2> / <float4>: scalar float tensor should resolve to vector TensorView
    ("func_tensorview_float2", _TorchTensor("float", 2), "TensorView<float2>", 1),
    ("func_tensorview_float4", _TorchTensor("float", 2), "TensorView<float4>", 1),
    # Non-float scalar should not resolve to TensorView<float2/float4>
    ("func_tensorview_float2", _TorchTensor("int", 2), None, None),
    ("func_tensorview_float4", _TorchTensor("int", 2), None, None),

    # DiffTensorView type resolution tests (CUDA-only, requires PyTorch)
    # NativeTorchTensorDiffPair should resolve to DiffTensorView<T>
    ("func_difftensorview_float", _TorchTensorDiffPair("float", 1), "DiffTensorView<float>", 1),
    ("func_difftensorview_float", _TorchTensorDiffPair("float", 2), "DiffTensorView<float>", 2),
    ("func_difftensorview_half", _TorchTensorDiffPair("half", 2), "DiffTensorView<half>", 2),

    # DiffTensorView float/half are both __BuiltinFloatingPointType, so resolution succeeds
    ("func_difftensorview_float", _TorchTensorDiffPair("half", 1), "DiffTensorView<float>", 1),
    ("func_difftensorview_half", _TorchTensorDiffPair("float", 1), "DiffTensorView<half>", 1),

    # Generic DiffTensorView<T> resolution
    ("func_difftensorview_generic", _TorchTensorDiffPair("float", 1), "DiffTensorView<float>", 1),
    ("func_difftensorview_generic", _TorchTensorDiffPair("half", 2), "DiffTensorView<half>", 2),
]

# fmt: on


def run_type_resolution_test(
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


def filter_tests(
    tests: List[Tuple[str, Any, Optional[str], Optional[int]]], types: Tuple[type, ...]
):
    a = []
    b = []
    for test in tests:
        # Exclude tests that use Tensor on non-supporting devices
        if isinstance(test[1], types):
            a.append(test)
        else:
            b.append(test)
    return a, b


SIMPLE_TESTS, TESTS = filter_tests(
    TESTS,
    types=(int, float),
)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name, arg_value, expected_type_name, expected_dim",
    SIMPLE_TESTS,
    ids=[f"{fn}_{av}" for fn, av, etn, ed in SIMPLE_TESTS],
)
def test_type_resolution_simple(
    device_type: spy.DeviceType,
    func_name: str,
    arg_value: Any,
    expected_type_name: Optional[str],
    expected_dim: Optional[int],
):
    run_type_resolution_test(device_type, func_name, arg_value, expected_type_name, expected_dim)


TENSOR_TESTS, TESTS = filter_tests(
    TESTS,
    types=(_Tensor,),
)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name, arg_value, expected_type_name, expected_dim",
    TENSOR_TESTS,
    ids=[f"{fn}_{av}" for fn, av, etn, ed in TENSOR_TESTS],
)
def test_type_resolution_tensor(
    device_type: spy.DeviceType,
    func_name: str,
    arg_value: Any,
    expected_type_name: Optional[str],
    expected_dim: Optional[int],
):
    run_type_resolution_test(device_type, func_name, arg_value, expected_type_name, expected_dim)


BUFFER_TESTS, TESTS = filter_tests(
    TESTS,
    types=(_Buffer,),
)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name, arg_value, expected_type_name, expected_dim",
    BUFFER_TESTS,
    ids=[f"{fn}_{av}" for fn, av, etn, ed in BUFFER_TESTS],
)
def test_type_resolution_buffer(
    device_type: spy.DeviceType,
    func_name: str,
    arg_value: Any,
    expected_type_name: Optional[str],
    expected_dim: Optional[int],
):
    run_type_resolution_test(device_type, func_name, arg_value, expected_type_name, expected_dim)


VECTOR_TESTS, TESTS = filter_tests(
    TESTS,
    types=(spy.int1, spy.float1, spy.int2, spy.float2, spy.int3, spy.float3, spy.float4),
)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name, arg_value, expected_type_name, expected_dim",
    VECTOR_TESTS,
    ids=[f"{fn}_{av}" for fn, av, etn, ed in VECTOR_TESTS],
)
def test_type_resolution_vector(
    device_type: spy.DeviceType,
    func_name: str,
    arg_value: Any,
    expected_type_name: Optional[str],
    expected_dim: Optional[int],
):
    run_type_resolution_test(device_type, func_name, arg_value, expected_type_name, expected_dim)


MATRIX_TESTS, TESTS = filter_tests(
    TESTS,
    types=(spy.float3x3, spy.float3x4, spy.float4x3, spy.float4x4),
)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name, arg_value, expected_type_name, expected_dim",
    MATRIX_TESTS,
    ids=[f"{fn}_{av}" for fn, av, etn, ed in MATRIX_TESTS],
)
def test_type_resolution_matrix(
    device_type: spy.DeviceType,
    func_name: str,
    arg_value: Any,
    expected_type_name: Optional[str],
    expected_dim: Optional[int],
):
    run_type_resolution_test(device_type, func_name, arg_value, expected_type_name, expected_dim)


TEXTURE_TESTS, TESTS = filter_tests(
    TESTS,
    types=(_Texture,),
)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name, arg_value, expected_type_name, expected_dim",
    TEXTURE_TESTS,
    ids=[f"{fn}_{av}" for fn, av, etn, ed in TEXTURE_TESTS],
)
def test_type_resolution_texture(
    device_type: spy.DeviceType,
    func_name: str,
    arg_value: Any,
    expected_type_name: Optional[str],
    expected_dim: Optional[int],
):
    if isinstance(arg_value, _Texture):
        if (
            arg_value.texture_type == spy.TextureType.texture_1d
            and device_type == spy.DeviceType.metal
        ):
            pytest.skip("Metal crashes testing 1D textures")
    run_type_resolution_test(device_type, func_name, arg_value, expected_type_name, expected_dim)


TORCHTENSOR_TESTS, TESTS = filter_tests(
    TESTS,
    types=(_TorchTensor, _TorchTensorDiffPair),
)

# TensorView requires CUDA device
CUDA_DEVICE_TYPES = (
    [spy.DeviceType.cuda] if spy.DeviceType.cuda in helpers.DEFAULT_DEVICE_TYPES else []
)


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
@pytest.mark.parametrize("device_type", CUDA_DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name, arg_value, expected_type_name, expected_dim",
    TORCHTENSOR_TESTS,
    ids=[f"{fn}_{av}" for fn, av, etn, ed in TORCHTENSOR_TESTS],
)
def test_type_resolution_tensorview(
    device_type: spy.DeviceType,
    func_name: str,
    arg_value: Any,
    expected_type_name: Optional[str],
    expected_dim: Optional[int],
):
    """Test TensorView type resolution with torch.Tensor arguments."""
    if not CUDA_DEVICE_TYPES:
        pytest.skip("CUDA device type not available")
    # Use torch device for TensorView tests
    device = helpers.get_torch_device(device_type)
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


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name, arg_value, expected_type_name, expected_dim",
    TESTS,
    ids=[f"{fn}_{av}" for fn, av, etn, ed in TESTS],
)
def test_type_resolution_other(
    device_type: spy.DeviceType,
    func_name: str,
    arg_value: Any,
    expected_type_name: Optional[str],
    expected_dim: Optional[int],
):
    run_type_resolution_test(device_type, func_name, arg_value, expected_type_name, expected_dim)
