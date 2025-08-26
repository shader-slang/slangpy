# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.helpers import test_id  # type: ignore (pytest fixture)


# TODO: Due to a bug in "Apple clang", the exception binding in nanobind
# raises RuntimeError instead of SlangCompileError
SlangCompileError = RuntimeError if sys.platform == "darwin" else spy.SlangCompileError


def create_session(device: spy.Device) -> spy.SlangSession:
    return device.create_slang_session(
        compiler_options={
            "include_paths": device.slang_session.desc.compiler_options.include_paths,
            "debug_info": spy.SlangDebugInfoLevel.standard,
        }
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_type_layout(test_id: str, device_type: spy.DeviceType):

    # Only do D3D for now
    if device_type != spy.DeviceType.d3d12:
        pytest.skip("Size / alignment checks only implemented for D3D at the moment")

    device = helpers.get_device(type=device_type)

    # Loading a valid module must succeed
    module = device.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        struct Foo {
            uint a;
            uint64_t b;
            uint16_t c;
            float d;
            uint16_t e;
            float4 f;
            uint64_t g;
            int4 h;
       };
        Foo foo;
        RWStructuredBuffer<Foo> foo_buffer;

        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main_a() {
            foo_buffer[0].a = 1;
        }
    """,
    )

    # Expected type layouts in constant buffer (size/align/offset)
    CBUFFER_EXPECTED_FIELDS = {
        "a": (4, 4, 0),
        "b": (8, 8, 8),
        "c": (2, 2, 16),
        "d": (4, 4, 20),
        "e": (2, 2, 24),
        "f": (16, 4, 32),
        "g": (8, 8, 48),
        "h": (16, 4, 64),
    }
    CBUFFER_EXPECTED_SIZE = 80

    # Expected type layouts in structured buffer (size/align/offset)
    SBUFFER_EXPECTED_FIELDS = {
        "a": (4, 4, 0),
        "b": (8, 8, 8),
        "c": (2, 2, 16),
        "d": (4, 4, 20),
        "e": (2, 2, 24),
        "f": (16, 4, 28),
        "g": (8, 8, 48),
        "h": (16, 4, 56),
    }
    SBUFFER_EXPECTED_SIZE = 72

    # Get the constant buffer.
    global_buffer_layout = module.layout.globals_type_layout
    constant_buffer_layout = global_buffer_layout.element_type_layout

    # Check 'foo' variable name on one line to verify nanobind
    # ownership doesn't destroy the field straight away.
    assert constant_buffer_layout.fields[0].name == "foo"

    # Get the 'foo' variable from the buffer.
    foo_variable = constant_buffer_layout.fields[0]
    assert foo_variable.name == "foo"
    foo_layout = foo_variable.type_layout
    assert foo_layout.name == "Foo"

    # Check type size and alignment.
    assert foo_layout.size == CBUFFER_EXPECTED_SIZE
    for field in foo_layout.fields:
        assert field.name in CBUFFER_EXPECTED_FIELDS
        assert field.type_layout.size == CBUFFER_EXPECTED_FIELDS[field.name][0]
        assert field.type_layout.alignment == CBUFFER_EXPECTED_FIELDS[field.name][1]
        assert field.offset == CBUFFER_EXPECTED_FIELDS[field.name][2]

    # Get the structured buffer and from there the type layout corresponding to the 'Foo' struct.
    structured_buffer_layout = constant_buffer_layout.fields[1]
    assert structured_buffer_layout.name == "foo_buffer"
    foo_layout = structured_buffer_layout.type_layout.unwrap_array()
    foo_layout = foo_layout.element_type_layout
    assert foo_layout.name == "Foo"

    # Check type size and alignment.
    assert foo_layout.size == SBUFFER_EXPECTED_SIZE
    for field in foo_layout.fields:
        assert field.name in SBUFFER_EXPECTED_FIELDS
        assert field.type_layout.size == SBUFFER_EXPECTED_FIELDS[field.name][0]
        assert field.type_layout.alignment == SBUFFER_EXPECTED_FIELDS[field.name][1]
        assert field.offset == SBUFFER_EXPECTED_FIELDS[field.name][2]


MODULE_SOURCE = """
        uniform int hello;
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void compute_main() {
        }
    """


@pytest.mark.skip("Crashes due to reflection cursor lifetime issues")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_cursor_lifetime_global_session(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    module = device.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=MODULE_SOURCE,
    )
    program = device.link_program([module], [module.entry_point("main")])

    # Get cursor for the program.
    cursor = spy.ReflectionCursor(program)

    # Clear all references to program, session and module.
    program = None
    module = None

    # Check what happens when accessing the cursor.
    assert cursor.has_field("hello")


@pytest.mark.skip("Crashes due to reflection cursor lifetime issues")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_cursor_lifetime_new_session(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module and program.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=MODULE_SOURCE,
    )
    program = session.link_program([module], [module.entry_point("main")])

    # Get cursor for the program.
    cursor = spy.ReflectionCursor(program)

    # Clear all references to program, session and module.
    program = None
    module = None
    session = None

    # Check what happens when accessing the cursor.
    assert cursor.has_field("hello")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_cursor_child_lifetime(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module and program.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=MODULE_SOURCE,
    )
    program = session.link_program([module], [module.entry_point("compute_main")])

    # Get cursor for the program and go straight to its child field.
    cursor = spy.ReflectionCursor(program).find_field("hello")

    # Clear all references to program, session and module.
    program = None
    module = None
    session = None

    # The child field should have kept the module alive.
    t = cursor.type
    assert t.name == "int"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_layout_lifetime(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=MODULE_SOURCE,
    )
    program_layout = module.layout

    # Clear all references to session and module.
    module = None
    session = None

    # Globals layout should still be valid, as it will have kept the module alive.
    globals_layout = program_layout.globals_type_layout
    assert globals_layout.element_type_layout.fields[0].name == "hello"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_declref_lifetime(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=MODULE_SOURCE,
    )
    module_decl = module.module_decl

    # Clear all references to session and module.
    module = None
    session = None

    # Decl ref should still be valid, as it will have kept the module alive.
    children = module_decl.children
    assert len(children) == 2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_declref_child_lifetime(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=MODULE_SOURCE,
    )
    var_decl = module.module_decl.children[0]

    # Clear all references to session and module.
    module = None
    session = None

    # Decl ref should still be valid, as it will have kept the module alive.
    assert var_decl.name == "hello"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_list_type_fields(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        struct MyType{
            int a;
            float b;
        };
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main() {
        }
    """,
    )

    # Get and read on 1 line.
    assert module.module_decl.children[0].as_type().fields[0].name == "a"
    assert module.module_decl.children[0].as_type().fields[1].name == "b"

    # By getting and storing as local.
    var_type = module.module_decl.children[0].as_type()
    fields = var_type.fields
    assert len(fields) == 2
    assert fields[0].name == "a"
    assert fields[1].name == "b"

    # Iterate.
    for field in fields:
        assert field.name in ["a", "b"]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_list_function_parameters(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        void test(int a, int b, int c) {
        }
    """,
    )

    # Get and read on 1 line.
    func = module.module_decl.children[0].as_function()
    assert func.parameters[0].name == "a"
    assert func.parameters[1].name == "b"
    assert func.parameters[2].name == "c"

    # Iterate.
    for p in func.parameters:
        assert p.name in ["a", "b", "c"]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_list_program_layout_params_and_entry_points(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        uniform int test;
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main1() {
        }
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main2() {
        }
    """,
    )
    program = session.link_program(
        [module], [module.entry_point("main1"), module.entry_point("main2")]
    )
    assert program.layout.parameters[0].name == "test"
    assert program.layout.entry_points[0].name == "main1"
    assert program.layout.entry_points[1].name == "main2"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_function_modifier(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        void test1(int a, int b, int c) {
        }
        [Differentiable]
        void test2(int a, int b, int c) {
        }
    """,
    )

    # Get and read on 1 line.
    func1 = module.module_decl.children[0].as_function()
    func2 = module.module_decl.children[1].as_function()
    assert func1.name == "test1"
    assert func2.name == "test2"
    assert not func1.has_modifier(spy.ModifierID.differentiable)
    assert func2.has_modifier(spy.ModifierID.differentiable)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_parameter_modifier(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        void test(inout int a, in int b, no_diff float c) {
        }
    """,
    )

    # Get and read on 1 line.
    func = module.module_decl.children[0].as_function()
    assert func.name == "test"
    assert func.parameters[0].has_modifier(spy.ModifierID.inout)
    assert func.parameters[1].has_modifier(spy.ModifierID.inn)
    assert func.parameters[2].has_modifier(spy.ModifierID.nodiff)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_value_modifier(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        no_diff int test(int a) {
        }
    """,
    )

    # Get and read on 1 line.
    func = module.module_decl.children[0].as_function()
    assert func.name == "test"
    assert func.has_modifier(spy.ModifierID.nodiff)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_full_type_name(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        struct A {
            struct B {
                float b_val;
            }
            float a_val;
        };

        struct ScalarTypes {
            bool bool_var;
            int32_t int32_var;
            uint32_t uint32_var;
            int64_t int64_var;
            uint64_t uint64_var;
            float16_t float16_var;
            float32_t float32_var;
            float64_t float64_var;
            int8_t int8_var;
            uint8_t uint8_var;
            int16_t int16_var;
            uint16_t uint16_var;
            int int_var;
            uint uint_var;
            half half_var;
            float float_var;
            double double_var;
        }

        struct VectorTypes {
            vector<float,3> vec_float_3_var;
            vector<float32_t,3> vec_float32_3_var;
            float3 float3_var;
        }
    """,
    )

    struct_a_decl = module.module_decl.find_first_child_of_kind(spy.DeclReflection.Kind.struct, "A")
    struct_a = struct_a_decl.as_type()
    assert struct_a.name == "A"
    assert struct_a.full_name == "A"

    struct_b = struct_a_decl.find_first_child_of_kind(spy.DeclReflection.Kind.struct, "B").as_type()
    assert struct_b.name == "B"
    assert struct_b.full_name == "A.B"

    scalar_types_decl = module.module_decl.find_first_child_of_kind(
        spy.DeclReflection.Kind.struct, "ScalarTypes"
    )
    scalar_types = scalar_types_decl.as_type()
    names = [(x.type.name, x.type.full_name, x.name) for x in scalar_types.fields]

    expected = [
        ("bool", "bool", "bool_var"),
        ("int", "int", "int32_var"),
        ("uint", "uint", "uint32_var"),
        ("int64_t", "int64_t", "int64_var"),
        ("uint64_t", "uint64_t", "uint64_var"),
        ("half", "half", "float16_var"),
        ("float", "float", "float32_var"),
        ("double", "double", "float64_var"),
        ("int8_t", "int8_t", "int8_var"),
        ("uint8_t", "uint8_t", "uint8_var"),
        ("int16_t", "int16_t", "int16_var"),
        ("uint16_t", "uint16_t", "uint16_var"),
        ("int", "int", "int_var"),
        ("uint", "uint", "uint_var"),
        ("half", "half", "half_var"),
        ("float", "float", "float_var"),
        ("double", "double", "double_var"),
    ]

    assert names == expected

    vector_types_decl = module.module_decl.find_first_child_of_kind(
        spy.DeclReflection.Kind.struct, "VectorTypes"
    )
    vector_types = vector_types_decl.as_type()
    names = [(x.type.name, x.type.full_name, x.name) for x in vector_types.fields]

    expected = [
        ("vector", "vector<float,3>", "vec_float_3_var"),
        ("vector", "vector<float,3>", "vec_float32_3_var"),
        ("vector", "vector<float,3>", "float3_var"),
    ]

    assert names == expected


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_null_type(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        struct A {
            float a_val;
        };
        uniform A myglobal;
    """,
    )

    globals_layout = module.layout.globals_type_layout
    globals_type_layout = globals_layout.element_type_layout
    invalid_element_type_layout = globals_type_layout.element_type_layout
    assert invalid_element_type_layout is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_deduplication(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        void funca(int p0, int p1) {}
        void funcb() {}
    """,
    )

    # Load the 2 functions, loading (a) twice.
    funca = module.module_decl.children[0].as_function()
    funcb = module.module_decl.children[1].as_function()
    funca_again = module.module_decl.children[0].as_function()

    # Check that the same function is returned.
    assert funca is funca_again
    assert funca is not funcb

    # Double check using explicit python object ids.
    assert id(funca) == id(funca_again)
    assert id(funca) != id(funcb)

    # Check that the parameters are the same.
    params_a = funca.parameters
    params_a_again = funca_again.parameters
    for p, p_again in zip([x for x in params_a], [y for y in params_a_again]):
        assert p is p_again


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_find_type_by_name(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        struct Hello {
            int a;
            void func2() {}
        }
    """,
    )

    # Find and verify the type and its internal function.
    t = module.layout.find_type_by_name("Hello")
    assert t is not None
    assert t.name == "Hello"
    f2 = module.layout.find_function_by_name_in_type(t, "func2")
    assert f2 is not None
    assert f2.name == "func2"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_find_generic_type_by_name(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        struct Hello<T> {
            T a;
            void func2() {}
        }
    """,
    )

    # Find and verify the type specialized with int.
    t = module.layout.find_type_by_name("Hello<int>")
    assert t is not None
    assert t.name == "Hello"
    a = t.fields[0]
    assert a.name == "a"
    assert a.type.name == "int"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_find_func_by_name(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int test()
        {
            return 0;
        }
    """,
    )

    # Find and verify the function.
    f1 = module.layout.find_function_by_name("test")
    assert f1 is not None
    assert f1.name == "test"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_get_type_layout(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        struct Hello {
            int a;
            void func2() {}
        }
        void test() {}
    """,
    )

    # Get type and convert to type layout.
    t = module.layout.find_type_by_name("Hello")
    assert t is not None
    assert t.name == "Hello"
    tl = module.layout.get_type_layout(t)
    assert tl is not None
    assert tl.name == "Hello"
    assert tl.size == 4


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_basic_function_overloads(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int foo()
        {
            return 0;
        }
        int bar()
        {
            return 0;
        }
        int bar(int i)
        {
            return 0;
        }
        int hi(float i)
        {
            return 0;
        }
        int hi(int i)
        {
            return 0;
        }
    """,
    )

    # Find and verify the function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0

    # Find and verify the function.
    f1 = module.layout.find_function_by_name("bar")
    assert f1 is not None
    assert f1.is_overloaded
    assert len(f1.overloads) == 2
    overloads = [x for x in f1.overloads]
    assert len(overloads[0].parameters) != len(overloads[1].parameters)

    # Find and verify the function.
    f1 = module.layout.find_function_by_name("hi")
    assert f1 is not None
    assert f1.is_overloaded
    assert len(f1.overloads) == 2
    assert len(f1.parameters) == 0
    overloads = [x for x in f1.overloads]
    assert len(overloads[0].parameters) == 1
    assert len(overloads[1].parameters) == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_type_method_overloads(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"test_type_method_overloads_{test_id}",
        source=r"""
        struct MyType {
            int foo() { return 0; }
            int foo(int a) { return a; }
        }
    """,
    )

    # Get MyType.
    mytype = module.layout.find_type_by_name("MyType")
    assert mytype is not None

    # Find and verify the function.
    f1 = module.layout.find_function_by_name_in_type(mytype, "foo")
    assert f1 is not None
    assert f1.is_overloaded
    assert len(f1.overloads) == 2
    overloads = [x for x in f1.overloads]
    l0 = len(overloads[0].parameters)
    l1 = len(overloads[1].parameters)
    assert l0 == 0 or l1 == 0
    assert l0 == 1 or l1 == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_type_constructor_overloads(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"test_type_constructor_overloads_{test_id}",
        source=r"""
        struct MyType {
            int val;
            __init() { this.val = 0; }
            __init(int a) { this.val = a; }
        }
    """,
    )

    # Get MyType.
    mytype = module.layout.find_type_by_name("MyType")
    assert mytype is not None

    # Find and verify the function.
    f1 = module.layout.find_function_by_name_in_type(mytype, "$init")
    assert f1 is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialize_concrete_function(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int foo(float3 val)
        {
            return 0;
        }
    """,
    )

    # Find and verify the generic function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0
    assert len(f1.parameters) == 1
    assert f1.parameters[0].name == "val"
    assert f1.parameters[0].type.full_name == "vector<float,3>"

    # Specialize the function with float3.
    f2 = f1.specialize_with_arg_types([module.layout.find_type_by_name("float3")])
    assert f2 is not None
    assert not f2.is_overloaded
    assert len(f2.overloads) == 0
    assert len(f2.parameters) == 1
    assert f2.parameters[0].name == "val"
    assert f2.parameters[0].type.full_name == "vector<float,3>"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialize_concrete_function_with_defaults(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int foo(float3 val = 0)
        {
            return 0;
        }
    """,
    )

    # Find and verify the generic function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0
    assert len(f1.parameters) == 1
    assert f1.parameters[0].name == "val"
    assert f1.parameters[0].type.full_name == "vector<float,3>"

    # Specialize the function with float3.
    f2 = f1.specialize_with_arg_types([])
    assert f2 is not None
    assert not f2.is_overloaded
    assert len(f2.overloads) == 0
    assert len(f2.parameters) == 1
    assert f2.parameters[0].name == "val"
    assert f2.parameters[0].type.full_name == "vector<float,3>"
    ""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialize_concrete_function_with_inout(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        void foo(inout float3 val)
        {
            val = 0;
        }
    """,
    )

    # Find and verify the generic function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0
    assert len(f1.parameters) == 1
    assert f1.parameters[0].name == "val"
    assert f1.parameters[0].type.full_name == "vector<float,3>"

    # Specialize the function with float3.
    f2 = f1.specialize_with_arg_types([module.layout.find_type_by_name("float3")])
    assert f2 is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fail_specialize_concrete_function(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int foo(float3 val)
        {
            return 0;
        }
    """,
    )

    # Find and verify the generic function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0
    assert len(f1.parameters) == 1
    assert f1.parameters[0].name == "val"
    assert f1.parameters[0].type.full_name == "vector<float,3>"

    # Fail to specialize the float3 function with float2.
    f2 = f1.specialize_with_arg_types([module.layout.find_type_by_name("float2")])
    assert f2 is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialize_interface_function(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int foo(IFloat val)
        {
            return 0;
        }
    """,
    )

    # Find and verify the generic function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0
    assert len(f1.parameters) == 1
    assert f1.parameters[0].name == "val"
    assert f1.parameters[0].type.name == "IFloat"

    # Specialize the function with float, still expect
    # to get back the interface type.
    f2 = f1.specialize_with_arg_types([module.layout.find_type_by_name("float")])
    assert f2 is not None
    assert not f2.is_overloaded
    assert len(f2.overloads) == 0
    assert len(f2.parameters) == 1
    assert f2.parameters[0].name == "val"
    assert f2.parameters[0].type.full_name == "IFloat"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialize_generic_function(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int foo<T>(T val)
        {
            return 0;
        }
    """,
    )

    # Find and verify the generic function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0
    assert len(f1.parameters) == 1
    assert f1.parameters[0].name == "val"
    assert f1.parameters[0].type.name == "T"

    # Specialize the function with float3.
    f2 = f1.specialize_with_arg_types([module.layout.find_type_by_name("float3")])
    assert f2 is not None
    assert not f2.is_overloaded
    assert len(f2.overloads) == 0
    assert len(f2.parameters) == 1
    assert f2.parameters[0].name == "val"
    assert f2.parameters[0].type.full_name == "vector<float,3>"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialize_partially_generic_function(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int foo<T>(T val, float3 val2)
        {
            return 0;
        }
    """,
    )

    # Find and verify the generic function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0
    assert len(f1.parameters) == 2
    assert f1.parameters[0].name == "val"
    assert f1.parameters[0].type.name == "T"
    assert f1.parameters[1].name == "val2"
    assert f1.parameters[1].type.full_name == "vector<float,3>"

    # Specialize the function with 2 float3s, where the first one is a generic.
    f2 = f1.specialize_with_arg_types(
        [
            module.layout.find_type_by_name("float3"),
            module.layout.find_type_by_name("float3"),
        ]
    )
    assert f2 is not None
    assert not f2.is_overloaded
    assert len(f2.overloads) == 0
    assert len(f2.parameters) == 2
    assert f2.parameters[0].name == "val"
    assert f2.parameters[0].type.full_name == "vector<float,3>"
    assert f2.parameters[1].name == "val2"
    assert f2.parameters[1].type.full_name == "vector<float,3>"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fail_specialize_partially_generic_function(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int foo<T>(T val, float3 val2)
        {
            return 0;
        }
    """,
    )

    # Find and verify the generic function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0
    assert len(f1.parameters) == 2
    assert f1.parameters[0].name == "val"
    assert f1.parameters[0].type.name == "T"
    assert f1.parameters[1].name == "val2"
    assert f1.parameters[1].type.full_name == "vector<float,3>"

    # Fail to specialize the function as the 2nd parameter is not a float2.
    f2 = f1.specialize_with_arg_types(
        [
            module.layout.find_type_by_name("float3"),
            module.layout.find_type_by_name("float2"),
        ]
    )
    assert f2 is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialize_partially_generic_function_with_implicit_cast(
    test_id: str, device_type: spy.DeviceType
):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=r"""
        int foo<T>(T val, float3 val2)
        {
            return 0;
        }
    """,
    )

    # Find and verify the generic function.
    f1 = module.layout.find_function_by_name("foo")
    assert f1 is not None
    assert not f1.is_overloaded
    assert len(f1.overloads) == 0
    assert len(f1.parameters) == 2
    assert f1.parameters[0].name == "val"
    assert f1.parameters[0].type.name == "T"
    assert f1.parameters[1].name == "val2"
    assert f1.parameters[1].type.full_name == "vector<float,3>"

    # Specialize the function with float3 and implicitly cast float.
    f2 = f1.specialize_with_arg_types(
        [
            module.layout.find_type_by_name("float3"),
            module.layout.find_type_by_name("float"),
        ]
    )
    assert f2 is not None
    assert not f2.is_overloaded
    assert len(f2.overloads) == 0
    assert len(f2.parameters) == 2
    assert f2.parameters[0].name == "val"
    assert f2.parameters[0].type.full_name == "vector<float,3>"
    assert f2.parameters[1].name == "val2"
    assert f2.parameters[1].type.full_name == "vector<float,3>"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_is_sub_type(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        interface IHello {
        }
        struct Hello : IHello {
            int a;
        }

        interface IHello2 {
        }
        extension Hello : IHello2 {
        }
    """,
    )

    t = module.layout.find_type_by_name("Hello")
    assert t is not None
    assert t.name == "Hello"

    # t should be a sub type of itself.
    assert module.layout.is_sub_type(t, t)

    # t should be a sub type of its interface.
    i = module.layout.find_type_by_name("IHello")
    assert i is not None
    assert i.name == "IHello"
    assert module.layout.is_sub_type(t, i)

    # t should be a sub type of its extension.
    i2 = module.layout.find_type_by_name("IHello2")
    assert i2 is not None
    assert i2.name == "IHello2"
    assert module.layout.is_sub_type(t, i)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hot_reload_invalid(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source=MODULE_SOURCE,
    )
    func = module.layout.find_function_by_name("compute_main")
    assert func is not None
    assert func.name == "compute_main"

    device.reload_all_programs()

    assert not func.is_valid

    with pytest.raises(RuntimeError, match="Reflection object has been invalidated"):
        x = func.name


# This test reproduces issues with reflection where 2 independently loaded
# modules can not resolve specialized versions of a generic function that
# is defined in one but not the other.
@pytest.mark.skip("Pending slang fix")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_independent_module_generic_resolution(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Create a session, and within it a module.
    session = create_session(device)

    generic_module = session.load_module_from_source(
        module_name=f"module_with_generic_{test_id}",
        source="""
T copy<T>(T val)
{
    return val;
}

struct GenericStruct<T>
{
    T val;
}

    """,
    )

    type_module = session.load_module_from_source(
        module_name=f"module_with_type_{test_id}",
        source="""
struct MyStruct
{
    int a;
}
    """,
    )

    # These first 4 versions don't currently work with slang

    find_func_via_generic_module = generic_module.layout.find_function_by_name("copy<MyStruct>")
    assert find_func_via_generic_module is not None

    find_func_via_type_module = type_module.layout.find_function_by_name("copy<MyStruct>")
    assert find_func_via_type_module is not None

    find_struct_via_generic_module = generic_module.layout.find_type_by_name(
        "GenericStruct<MyStruct>"
    )
    assert find_struct_via_generic_module is not None

    find_struct_via_type_module = type_module.layout.find_type_by_name("GenericStruct<MyStruct>")
    assert find_struct_via_type_module is not None

    # This does currently work
    mystruct = type_module.layout.find_type_by_name("MyStruct")
    gen_func = generic_module.layout.find_function_by_name("copy")
    spec_func_via_generic_module = gen_func.specialize_with_arg_types([mystruct])
    assert spec_func_via_generic_module is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
