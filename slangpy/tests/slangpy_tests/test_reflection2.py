# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from slangpy import DeviceType, Tensor, TextureUsage, TypeReflection
import slangpy.reflection as r
from slangpy.core.enums import IOType
from slangpy.core.function import Function
from slangpy.native_func import BaseModule
from slangpy.reflection.lookup import resolve_element_type, resolve_program_layout
from slangpy.testing import helpers

from typing import Any, Callable

MODULE = """
import "slangpy";
float foo(float a) { return a; }
float foo2(float a, float b) { return a+b; }
float foo_v3(float3 a) { return a.x; }
float foo_ol(float a) { return a; }
float foo_ol(float a, float b) { return a+b; }
float foo_generic<T>(T a) { return 0; }
struct Foo
{
    float3 value;
    int bar(float a) { return 0; }
}

struct GenericType<A, int N> {}
struct BoolGenericType<let Enabled: bool> {}

void update(inout float value, out float result, no_diff in float weight) {
    result = value * weight;
}

void use_textures(Texture2D<float4> texture, RWTexture2D<float4> rw_texture) {}

"""


FLOAT_SCALAR_TYPES = {
    TypeReflection.ScalarType.float16,
    TypeReflection.ScalarType.float32,
    TypeReflection.ScalarType.float64,
}


def is_float(kind: TypeReflection.ScalarType) -> bool:
    return kind in FLOAT_SCALAR_TYPES


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vector_types_correct(device_type: DeviceType):
    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)
    layout = function.module.layout

    for st in TypeReflection.ScalarType:
        if st == TypeReflection.ScalarType.void or st == TypeReflection.ScalarType.none:
            continue
        for i in range(1, 4):
            assert layout.vector_type(st, i).num_elements == i


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_matrix_types_correct(device_type: DeviceType):
    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)
    layout = function.module.layout

    for st in TypeReflection.ScalarType:
        if st == TypeReflection.ScalarType.void or st == TypeReflection.ScalarType.none:
            continue
        for row in range(1, 4):
            for col in range(1, 4):
                m = layout.matrix_type(st, row, col)
                assert m.rows == row
                assert m.cols == col


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_basic_function_decl(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo2", MODULE)
    layout = function.module.layout

    res = layout.find_function_by_name("foo2")
    assert res is not None
    assert res.name == "foo2"
    assert res.parameters[0].name == "a"
    assert res.parameters[0].type == layout.scalar_type(TypeReflection.ScalarType.float32)
    assert res.parameters[1].name == "b"
    assert res.parameters[1].type == layout.scalar_type(TypeReflection.ScalarType.float32)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_method(device_type: DeviceType):

    device = helpers.get_device(device_type)
    m = helpers.create_module(device, MODULE)
    layout = m.layout

    struct = m.find_struct("Foo")
    assert struct is not None

    func = m.find_function_in_struct(struct, "bar")
    assert func is not None
    assert isinstance(func, Function)
    res = func._slang_func
    assert res.name == "bar"
    assert res.parameters[0].name == "a"
    assert res.parameters[0].type == layout.scalar_type(TypeReflection.ScalarType.float32)
    assert res.have_return_value
    assert res.return_type == layout.scalar_type(TypeReflection.ScalarType.int32)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_and_struct_reflection_storage(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    assert not hasattr(module, "module")
    assert "layout" not in module.__dict__
    assert "device_module" not in module.__dict__
    assert module.layout.find_type_by_name("Foo") is not None

    struct = module.Foo.as_struct()

    assert struct.module is module
    assert struct.struct is struct.type
    assert "module" not in struct.__dict__
    assert "struct" not in struct.__dict__
    assert struct.struct.type_reflection.full_name == "Foo"
    assert struct.type_reflection.full_name == "Foo"
    assert not hasattr(struct.struct, "reflection")
    assert not hasattr(struct, "reflection")

    tensor = Tensor.empty(device, (2,), dtype=struct)
    assert tensor.dtype is struct.struct


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reflection_layout_tracks_hot_reload_generation(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)
    layout = r.SlangProgramLayout(module.device_module.layout)

    generation = layout.generation
    layout.on_hot_reload(module.device_module.layout)
    module.on_hot_reload()

    assert layout.generation == generation + 1
    assert module.layout.find_type_by_name("Foo") is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_hot_reload_refreshes_existing_reflection_objects(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module_a = device.load_module_from_source(
        "reflection_hot_reload_a",
        """
struct Foo { float value; };
float get_value(Foo foo) { return foo.value; }
""",
    )
    module_b = device.load_module_from_source(
        "reflection_hot_reload_b",
        """
struct Foo { int value; };
int get_value(Foo foo) { return foo.value; }
""",
    )
    module = r.SlangProgramLayout(module_a.layout)

    old_foo = module.require_type_by_name("Foo")
    old_function = module.require_function_by_name("get_value")

    module.on_hot_reload(module_b.layout)

    assert module.require_type_by_name("Foo") is old_foo
    assert old_foo.fields["value"].type is module.scalar_type(TypeReflection.ScalarType.int32)
    assert module.require_function_by_name("get_value") is old_function
    assert old_function.return_type is module.scalar_type(TypeReflection.ScalarType.int32)

    wrapped_module = helpers.create_module(device, "float get_value(float value) { return value; }")
    function = wrapped_module.get_value
    BaseModule.on_hot_reload(wrapped_module, module_b, module_b.layout)
    assert function._slang_func.return_type is wrapped_module.layout.scalar_type(
        TypeReflection.ScalarType.int32
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_known_type_helpers(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)
    layout = module.layout

    unknown_type = layout.require_type_by_name("Unknown")
    float_type = layout.require_type_by_name("float")

    assert r.is_unknown(unknown_type)
    assert not r.is_known(unknown_type)
    assert not r.is_known_or_none(unknown_type)

    assert not r.is_unknown(float_type)
    assert r.is_known(float_type)
    assert r.is_known_or_none(float_type)

    assert not r.is_unknown(None)
    assert r.is_known_or_none(None)
    with pytest.raises(ValueError, match="Type is None"):
        r.is_known(None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reflection_layout_creates_native_types(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)
    layout = r.SlangProgramLayout(module.device_module.layout)

    float_type = layout.require_type_by_name("float")
    assert isinstance(float_type, r.ScalarType)
    assert float_type.full_name == "float"
    assert float_type.shape == ()

    void_type = layout.scalar_type(TypeReflection.ScalarType.void)
    assert isinstance(void_type, r.VoidType)
    assert void_type.full_name == "void"

    vector_type = layout.require_type_by_name("vector<float,3>")
    assert isinstance(vector_type, r.VectorType)
    assert vector_type.element_type is float_type
    assert vector_type.num_elements == 3
    assert vector_type.shape == (3,)

    matrix_type = layout.require_type_by_name("matrix<float,3,2>")
    assert isinstance(matrix_type, r.MatrixType)
    assert matrix_type.inner_element_type is float_type
    assert matrix_type.shape == (3, 2)

    array_type = layout.require_type_by_name("float[4]")
    assert isinstance(array_type, r.ArrayType)
    assert array_type.element_type is float_type
    assert array_type.array_shape == (4,)

    struct_type = layout.require_type_by_name("Foo")
    assert isinstance(struct_type, r.StructType)
    assert struct_type.full_name == "Foo"
    assert isinstance(struct_type.fields["value"], r.SlangField)
    assert struct_type.fields["value"].type is vector_type

    tensor_type = layout.tensor_type(
        float_type,
        2,
        access=r.TensorAccess.read_write,
        tensor_kind=r.TensorType.tensor,
    )
    assert isinstance(tensor_type, r.ITensorType)
    assert tensor_type.dtype is float_type
    assert tensor_type.dims == 2
    assert tensor_type.readable
    assert tensor_type.writable
    assert tensor_type.shape == (-1, -1)

    texture_function = layout.require_function_by_name("use_textures")
    texture_type = texture_function.parameters[0].type
    rw_texture_type = texture_function.parameters[1].type
    assert isinstance(texture_type, r.TextureType)
    assert texture_type.usage == TextureUsage.shader_resource
    assert isinstance(rw_texture_type, r.TextureType)
    assert rw_texture_type.usage == TextureUsage.unordered_access


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reflection_layout_creates_function_metadata(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)
    layout = r.SlangProgramLayout(module.device_module.layout)

    float_type = layout.require_type_by_name("float")
    struct_type = layout.require_type_by_name("Foo")

    function = layout.require_function_by_name("foo2")
    assert isinstance(function, r.SlangFunction)
    assert function.name == "foo2"
    assert function.return_type is float_type
    assert function.have_return_value

    parameters = function.parameters
    assert len(parameters) == 2
    assert isinstance(parameters[0], r.SlangParameter)
    assert parameters[0].name == "a"
    assert parameters[0].index == 0
    assert parameters[0].type is float_type
    assert parameters[0].io_type == IOType.inn

    update = layout.require_function_by_name("update")
    assert update.parameters[0].io_type == IOType.inout
    assert update.parameters[1].io_type == IOType.out
    assert update.parameters[2].no_diff

    method = layout.require_function_by_name_in_type(struct_type, "bar")
    assert method.this_type is struct_type
    assert method.full_name == "bar"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reflection_lookup_resolves_builtin_and_struct_types(device_type: DeviceType):
    device = helpers.get_device(device_type)

    builtin_layout = resolve_program_layout(device, None, None)
    assert builtin_layout.find_type_by_name("float") is not None
    assert resolve_program_layout(device, "float", None) is builtin_layout

    float_type = resolve_element_type(builtin_layout, "float")
    assert isinstance(float_type, r.ScalarType)
    assert float_type.full_name == "float"
    assert resolve_element_type(builtin_layout, int).full_name == "int"

    generation = builtin_layout.generation
    device.reload_all_programs()
    assert builtin_layout.generation == generation + 1

    module = helpers.create_module(device, MODULE)
    struct = module.Foo.as_struct()
    struct_layout = resolve_program_layout(device, struct, None)
    assert struct_layout.find_type_by_name("Foo") is not None

    python_struct_type = module.layout.require_type_by_name("Foo")
    assert resolve_element_type(struct_layout, struct).full_name == "Foo"
    assert resolve_element_type(struct_layout, python_struct_type).full_name == "Foo"
    assert (
        resolve_element_type(struct_layout, python_struct_type.buffer_layout.reflection).full_name
        == "Foo"
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_specialization(device_type: DeviceType):
    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, "foo_generic<float>", MODULE)
    assert function is not None

    layout = function.module.layout

    res = layout.find_function_by_name("foo_generic<float>")
    assert res is not None
    assert res.name == "foo_generic"
    assert res.full_name == "foo_generic<float>"
    assert res.parameters[0].name == "a"
    assert res.parameters[0].type == layout.scalar_type(TypeReflection.ScalarType.float32)

    generic = layout.find_function_by_name("foo_generic")
    assert generic is not None
    specialized = generic.specialize_with_arg_types(
        [layout.scalar_type(TypeReflection.ScalarType.float32)]
    )
    assert specialized.name == "foo_generic"
    assert specialized.return_type is layout.scalar_type(TypeReflection.ScalarType.float32)
    assert layout.find_function_by_name("foo_generic") is generic
    assert layout.find_function_by_name("foo_generic<float>") is res
    assert specialized is not res


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_parsing(device_type: DeviceType):
    device = helpers.get_device(device_type)
    m = helpers.create_module(device, MODULE)
    layout = m.layout

    generic = layout.find_type_by_name("GenericType<GenericType<GenericType<float, 1>, 2>, 3>")
    assert generic is not None

    args = layout.get_resolved_generic_args(generic)
    assert args is not None
    assert len(args) == 2
    assert isinstance(args[0], r.SlangType)
    assert args[0].full_name == "GenericType<GenericType<float, 1>, 2>"
    assert isinstance(args[1], int)
    assert args[1] == 3


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    ("type_name", "expected_value"),
    [("BoolGenericType<true>", 1), ("BoolGenericType<false>", 0)],
)
def test_bool_generic_value_parsing(device_type: DeviceType, type_name: str, expected_value: int):
    device = helpers.get_device(device_type)
    m = helpers.create_module(device, MODULE)
    layout = m.layout

    generic = layout.find_type_by_name(type_name)
    assert generic is not None

    args = layout.get_resolved_generic_args(generic)
    assert args is not None
    assert len(args) == 1
    assert isinstance(args[0], int)
    assert args[0] == expected_value


def check_texture(
    type: r.SlangType,
    resource_shape: TypeReflection.ResourceShape,
    resource_access: TypeReflection.ResourceAccess,
    num_dims: int,
    element_type: str,
):
    assert isinstance(type, r.TextureType)
    assert type.resource_shape == resource_shape
    assert type.resource_access == resource_access
    assert type.texture_dims == num_dims

    et = type.program.find_type_by_name(element_type)
    assert et is not None
    assert type.num_dims == type.texture_dims + et.num_dims
    assert type.element_type == et


def check_scalar(type: r.SlangType, scalar_type: TypeReflection.ScalarType):
    assert isinstance(type, r.ScalarType)
    assert type.slang_scalar_type == scalar_type
    assert type.differentiable == is_float(scalar_type)


def check_vector(type: r.SlangType, scalar_type: TypeReflection.ScalarType, size: int):
    assert isinstance(type, r.VectorType)
    assert isinstance(type.element_type, r.ScalarType)
    assert type.element_type.slang_scalar_type == scalar_type
    assert type.num_elements == size
    assert type.differentiable == type.element_type.differentiable


def check_matrix(type: r.SlangType, scalar_type: TypeReflection.ScalarType, rows: int, cols: int):
    assert isinstance(type, r.MatrixType)
    assert isinstance(type.element_type, r.VectorType)
    assert type.rows == rows
    assert type.cols == cols
    assert type.differentiable == type.element_type.differentiable


def check_structured_buffer(
    type: r.SlangType, resource_access: TypeReflection.ResourceAccess, element_type: str
):
    assert isinstance(type, r.StructuredBufferType)
    assert type.element_type == type.program.find_type_by_name(element_type)
    assert type.resource_access == resource_access


def check_address_buffer(type: r.SlangType, resource_access: TypeReflection.ResourceAccess):
    assert isinstance(type, r.ByteAddressBufferType)
    assert type.element_type == type.program.find_type_by_name("uint8_t")
    assert type.resource_access == resource_access


def check_array(type: r.SlangType, element_type: str, num_elements: int):
    assert isinstance(type, r.ArrayType)
    assert type.element_type is not None
    assert type.element_type == type.program.find_type_by_name(element_type)
    assert type.num_elements == num_elements
    if num_elements == 0:
        assert type.full_name == f"{type.element_type.full_name}[]"
    else:
        assert type.full_name == f"{type.element_type.full_name}[{num_elements}]"
    assert type.differentiable == type.element_type.differentiable


def check_struct(type: r.SlangType, fields: dict[str, str]):
    assert isinstance(type, r.StructType)

    input_field_types = {n: type.program.find_type_by_name(t) for (n, t) in fields.items()}
    struct_field_types = {f.name: f.type for f in type.fields.values()}
    assert input_field_types == struct_field_types


def check_interface(type: r.SlangType):
    assert isinstance(type, r.InterfaceType)


ARG_TYPE_CHECKS = [
    ("float16_t", lambda x: check_scalar(x, TypeReflection.ScalarType.float16)),
    ("float", lambda x: check_scalar(x, TypeReflection.ScalarType.float32)),
    ("int8_t", lambda x: check_scalar(x, TypeReflection.ScalarType.int8)),
    ("int16_t", lambda x: check_scalar(x, TypeReflection.ScalarType.int16)),
    ("int", lambda x: check_scalar(x, TypeReflection.ScalarType.int32)),
    ("int64_t", lambda x: check_scalar(x, TypeReflection.ScalarType.int64)),
    ("uint8_t", lambda x: check_scalar(x, TypeReflection.ScalarType.uint8)),
    ("uint16_t", lambda x: check_scalar(x, TypeReflection.ScalarType.uint16)),
    ("uint", lambda x: check_scalar(x, TypeReflection.ScalarType.uint32)),
    ("uint64_t", lambda x: check_scalar(x, TypeReflection.ScalarType.uint64)),
    ("float3", lambda x: check_vector(x, TypeReflection.ScalarType.float32, 3)),
    ("float4", lambda x: check_vector(x, TypeReflection.ScalarType.float32, 4)),
    (
        "vector<float,4>",
        lambda x: check_vector(x, TypeReflection.ScalarType.float32, 4),
    ),
    ("int3", lambda x: check_vector(x, TypeReflection.ScalarType.int32, 3)),
    ("bool2", lambda x: check_vector(x, TypeReflection.ScalarType.bool, 2)),
    ("uint1", lambda x: check_vector(x, TypeReflection.ScalarType.uint32, 1)),
    ("float3x4", lambda x: check_matrix(x, TypeReflection.ScalarType.float32, 3, 4)),
    (
        "matrix<float,3,4>",
        lambda x: check_matrix(x, TypeReflection.ScalarType.float32, 3, 4),
    ),
    (
        "Texture1D<float>",
        lambda x: check_texture(
            x,
            TypeReflection.ResourceShape.texture_1d,
            TypeReflection.ResourceAccess.read,
            1,
            "float",
        ),
    ),
    (
        "RWTexture1D<float>",
        lambda x: check_texture(
            x,
            TypeReflection.ResourceShape.texture_1d,
            TypeReflection.ResourceAccess.read_write,
            1,
            "float",
        ),
    ),
    (
        "Texture2D<float3>",
        lambda x: check_texture(
            x,
            TypeReflection.ResourceShape.texture_2d,
            TypeReflection.ResourceAccess.read,
            2,
            "float3",
        ),
    ),
    (
        "RWTexture2D<float3>",
        lambda x: check_texture(
            x,
            TypeReflection.ResourceShape.texture_2d,
            TypeReflection.ResourceAccess.read_write,
            2,
            "float3",
        ),
    ),
    (
        "StructuredBuffer<float>",
        lambda x: check_structured_buffer(x, TypeReflection.ResourceAccess.read, "float"),
    ),
    (
        "RWStructuredBuffer<float4>",
        lambda x: check_structured_buffer(x, TypeReflection.ResourceAccess.read_write, "float4"),
    ),
    ("float[10]", lambda x: check_array(x, "float", 10)),
    ("float3[]", lambda x: check_array(x, "float3", 0)),
    (
        "ByteAddressBuffer",
        lambda x: check_address_buffer(x, TypeReflection.ResourceAccess.read),
    ),
    (
        "RWByteAddressBuffer",
        lambda x: check_address_buffer(x, TypeReflection.ResourceAccess.read_write),
    ),
    ("TestStruct", lambda x: check_struct(x, {"foo": "float"})),
    ("ITestInterface", lambda x: check_interface(x)),
]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("arg_type", ARG_TYPE_CHECKS, ids=lambda x: x[0])
def test_arg_types(device_type: DeviceType, arg_type: tuple[str, Callable[[Any], bool]]):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "foo",
        f"""
import "slangpy";
struct TestStruct {{
    float foo;
}}
interface ITestInterface {{}}

float foo({arg_type[0]} a) {{ return 0; }}
""",
    )
    layout = function.module.layout

    res = layout.find_function_by_name("foo")
    assert res is not None
    assert res.name == "foo"
    assert res.parameters[0].name == "a"

    arg_type[1](res.parameters[0].type)


def compare_struct_values(refl_val: Any, spy_val: Any):
    if isinstance(refl_val, r.SlangType):
        assert refl_val == spy_val.struct
    elif isinstance(refl_val, list):
        assert len(refl_val) == len(spy_val)
        for i in range(len(refl_val)):
            compare_struct_values(refl_val[i], spy_val[i])
    elif isinstance(refl_val, dict):
        assert len(refl_val) == len(spy_val)
        for key in refl_val:
            compare_struct_values(refl_val[key], spy_val[key])
    else:
        assert refl_val == spy_val


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_refl_duck_typing(device_type: DeviceType):

    device = helpers.get_device(device_type)
    module = helpers.create_module(
        device,
        """
struct MyStruct: IDifferentiable {
    int a;
}
""",
    )
    layout = module.layout

    refl_struct = layout.find_type_by_name("MyStruct")
    assert refl_struct is not None
    assert isinstance(refl_struct, r.StructType)
    assert refl_struct.name == "MyStruct"

    spy_struct = module.MyStruct.as_struct()
    assert spy_struct is not None

    fields = [x for x in dir(refl_struct) if not x.startswith("_")]
    for field in fields:
        refl_val = getattr(refl_struct, field)

        # ignore if attribute is a function
        if callable(refl_val):
            continue

        spy_val = getattr(spy_struct, field)
        compare_struct_values(refl_val, spy_val)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("runa", [False, True])
@pytest.mark.parametrize("runb", [False, True])
def test_interface(device_type: DeviceType, runa: bool, runb: bool):

    # Note: use_cache=False to avoid any caching effects interfering with separate test runs
    device = helpers.get_device(type=device_type, use_cache=False)

    # Create a module with an IFoo interface, a Foo struct implementing it, and a function taking IFoo
    code = f"""
    interface IFoo {{}}
    struct Foo: IFoo {{}}
    void test_func(IFoo x) {{}}
    """
    module1 = helpers.create_module(device, code)

    # If enabled, attempt to specialize test_func with Foo
    if runa:
        func = module1.layout.require_function_by_name("test_func").specialize_with_arg_types(
            [module1.layout.require_type_by_name("Foo")]
        )
        assert func is not None, "Could not specialize function 1"

    # Create a 2nd similar module with IFoo2, Foo2, and test_func2. In this case, Foo2 implements IFoo2 via an extension.
    code = f"""
    interface IFoo2 {{}}
    struct Foo2 {{}}
    extension Foo2: IFoo2 {{}}
    void test_func2(IFoo2 x) {{}}
    """
    module2 = helpers.create_module(device, code)

    # If enabled, attempt to specialize test_func2 with Foo2
    # This fails if runa is also True
    if runb:
        func = module2.layout.require_function_by_name("test_func2").specialize_with_arg_types(
            [module2.layout.require_type_by_name("Foo2")]
        )
        assert func is not None, "Could not specialize function 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
