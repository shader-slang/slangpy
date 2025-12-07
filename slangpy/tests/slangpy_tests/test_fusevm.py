# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for the bytecode-based fusion VM system.
"""

import slangpy as spy
import slangpy.testing.helpers as helpers
from slangpy.experimental.fusevm import (
    FuseProgram,
    FuseProgramBuilder,
    OpCode,
    CallSlangInstruction,
    CallSubInstruction,
)


def create_bind_context(module):
    """
    Helper function to create a BindContext for testing.

    Args:
        module: The Slang module

    Returns:
        A BindContext configured for primitive mode
    """
    from slangpy.bindings.marshall import BindContext
    from slangpy.core.native import CallMode, CallDataMode

    return BindContext(
        module.layout,
        CallMode.prim,
        module.device_module,
        {},
        CallDataMode.global_data,
    )


def test_basic_program_creation():
    """Test creating an empty program."""
    program = FuseProgram("test_program")

    assert program.name == "test_program"
    assert len(program.instructions) == 0
    assert len(program.variables) == 0
    assert len(program.input_vars) == 0
    assert len(program.output_vars) == 0


def test_variable_allocation():
    """Test allocating variables."""
    program = FuseProgram("test_vars")

    # Allocate some variables
    var0 = program.allocate_variable("x")
    var1 = program.allocate_variable("y")
    var2 = program.allocate_variable("result")

    assert var0 == 0
    assert var1 == 1
    assert var2 == 2

    assert len(program.variables) == 3
    assert program.get_variable(0).name == "x"
    assert program.get_variable(1).name == "y"
    assert program.get_variable(2).name == "result"


def test_input_output_tracking():
    """Test marking variables as inputs/outputs."""
    program = FuseProgram("test_io")

    var_a = program.allocate_variable("a")
    var_b = program.allocate_variable("b")
    var_result = program.allocate_variable("result")

    program.add_input(var_a)
    program.add_input(var_b)
    program.add_output(var_result)

    assert program.input_vars == [0, 1]
    assert program.output_vars == [2]


def test_program_builder():
    """Test the fluent builder interface."""
    builder = FuseProgramBuilder("test_builder")

    # Build a simple program structure
    a = builder.input("a")
    b = builder.input("b")
    temp = builder.temp("temp")
    result = builder.output("result")

    program = builder.build()

    assert program.name == "test_builder"
    assert len(program.variables) == 4
    assert program.input_vars == [0, 1]
    assert program.output_vars == [3]
    assert program.get_variable(a).name == "a"
    assert program.get_variable(b).name == "b"
    assert program.get_variable(temp).name == "temp"
    assert program.get_variable(result).name == "result"


def test_call_slang_instruction():
    """Test creating a CALL_SLANG instruction."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    builder = FuseProgramBuilder("test_call_slang")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")

    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()

    assert len(program.instructions) == 1
    instr = program.instructions[0]
    assert isinstance(instr, CallSlangInstruction)
    assert instr.opcode == OpCode.CALL_SLANG
    assert instr.function == ft_add
    assert instr.inputs == [0, 1]
    assert instr.outputs == [2]


def test_call_sub_instruction():
    """Test creating a CALL_SUB instruction."""
    # Create a sub-program
    sub_builder = FuseProgramBuilder("sub_func")
    sub_x = sub_builder.input("x")
    sub_y = sub_builder.input("y")
    sub_out = sub_builder.output("out")
    sub_program = sub_builder.build()

    # Create main program that calls the sub-program
    builder = FuseProgramBuilder("test_call_sub")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")

    builder.call_sub(sub_program, [a, b], [result])

    program = builder.build()

    assert len(program.instructions) == 1
    instr = program.instructions[0]
    assert isinstance(instr, CallSubInstruction)
    assert instr.opcode == OpCode.CALL_SUB
    assert instr.sub_program == sub_program
    assert instr.inputs == [0, 1]
    assert instr.outputs == [2]


def test_program_dump():
    """Test that the dump method produces readable output."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    builder = FuseProgramBuilder("test_dump")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()
    dump = program.dump()

    # Check that key information appears in the dump
    assert "test_dump" in dump
    assert "v0:a" in dump
    assert "v1:b" in dump
    assert "v2:result" in dump
    assert "CALL_SLANG" in dump
    assert "ft_add" in dump

    # Print it for manual inspection
    print("\n" + dump)


# ============================================================================
# Step 3: More Complex Tests (similar to test_fusion.py)
# ============================================================================


def test_simple_call_single_function():
    """Test calling a single Slang function: result = ft_add(a, b)"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    # Build program: result = ft_add(a, b)
    builder = FuseProgramBuilder("simple_add")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()

    # Verify structure
    assert len(program.instructions) == 1
    assert len(program.variables) == 3
    assert program.input_vars == [a, b]
    assert program.output_vars == [result]

    print("\n" + program.dump())


def test_two_sequential_calls():
    """Test calling two functions sequentially: temp = ft_mul(a, b); result = ft_add(temp, c)"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Build program: temp = ft_mul(a, b); result = ft_add(temp, c)
    builder = FuseProgramBuilder("mul_then_add")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_slang(ft_add, [temp, c], [result])

    program = builder.build()

    # Verify structure
    assert len(program.instructions) == 2
    assert len(program.variables) == 5
    assert program.input_vars == [a, b, c]
    assert program.output_vars == [result]

    # Check instruction sequence
    assert isinstance(program.instructions[0], CallSlangInstruction)
    assert program.instructions[0].function.name == "ft_mul"
    assert program.instructions[0].inputs == [a, b]
    assert program.instructions[0].outputs == [temp]

    assert isinstance(program.instructions[1], CallSlangInstruction)
    assert program.instructions[1].function.name == "ft_add"
    assert program.instructions[1].inputs == [temp, c]
    assert program.instructions[1].outputs == [result]

    print("\n" + program.dump())


def test_three_calls_with_dependency():
    """Test chain of three calls: (a * b) + c -> final = add(mul_result, c)"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")
    ft_return1 = module.require_function("ft_return1")

    # Build program: temp1 = mul(a, b); temp2 = add(temp1, c); result = return1(temp2)
    builder = FuseProgramBuilder("chain_three")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp1])
    builder.call_slang(ft_add, [temp1, c], [temp2])
    builder.call_slang(ft_return1, [temp2], [result])

    program = builder.build()

    # Verify structure
    assert len(program.instructions) == 3
    assert len(program.variables) == 6
    assert program.input_vars == [a, b, c]
    assert program.output_vars == [result]

    print("\n" + program.dump())


def test_independent_operations():
    """Test two independent operations that don't depend on each other."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")
    ft_mul = module.require_function("ft_mul")

    # Build program with two independent operations
    # temp1 = add(a, b)
    # temp2 = mul(c, d)
    # result = add(temp1, temp2)
    builder = FuseProgramBuilder("independent_ops")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_add, [a, b], [temp1])
    builder.call_slang(ft_mul, [c, d], [temp2])
    builder.call_slang(ft_add, [temp1, temp2], [result])

    program = builder.build()

    # Verify structure
    assert len(program.instructions) == 3
    assert len(program.variables) == 7

    print("\n" + program.dump())


def test_nested_sub_program():
    """Test calling a sub-program within a main program."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")
    ft_mul = module.require_function("ft_mul")

    # Create a sub-program that adds two numbers
    sub_builder = FuseProgramBuilder("add_sub")
    sub_x = sub_builder.input("x")
    sub_y = sub_builder.input("y")
    sub_result = sub_builder.output("result")
    sub_builder.call_slang(ft_add, [sub_x, sub_y], [sub_result])
    sub_program = sub_builder.build()

    # Create main program that uses the sub-program
    # temp1 = mul(a, b)
    # result = add_sub(temp1, c)  # Calls the sub-program
    builder = FuseProgramBuilder("main_with_sub")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp1 = builder.temp("temp1")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp1])
    builder.call_sub(sub_program, [temp1, c], [result])

    program = builder.build()

    # Verify structure
    assert len(program.instructions) == 2
    assert isinstance(program.instructions[0], CallSlangInstruction)
    assert isinstance(program.instructions[1], CallSubInstruction)
    assert program.instructions[1].sub_program == sub_program

    print("\n" + program.dump())


def test_reusing_same_function():
    """Test calling the same Slang function multiple times."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    # Build program that calls ft_add three times
    # temp1 = add(a, b)
    # temp2 = add(c, d)
    # result = add(temp1, temp2)
    builder = FuseProgramBuilder("reuse_add")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_add, [a, b], [temp1])
    builder.call_slang(ft_add, [c, d], [temp2])
    builder.call_slang(ft_add, [temp1, temp2], [result])

    program = builder.build()

    # Verify all three instructions use the same function
    assert len(program.instructions) == 3
    assert all(instr.function.name == "ft_add" for instr in program.instructions)

    print("\n" + program.dump())


# ============================================================================
# Step 4: Type Inference Tests
# ============================================================================


def test_type_inference_single_function():
    """Test type inference for a single function call."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    # Build program without specifying types
    builder = FuseProgramBuilder("test_type_inference")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()

    # Before type inference, types should be None
    assert program.get_variable(a).slang is None
    assert program.get_variable(b).slang is None
    assert program.get_variable(result).slang is None

    # Run type inference with bind context and input marshals
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 10, 20)
    success = program.infer_types(context, bindings.args)
    assert success, "Type inference should succeed"

    # After type inference, all variables should have types
    var_a = program.get_variable(a)
    var_b = program.get_variable(b)
    var_result = program.get_variable(result)

    assert var_a.slang is not None, "Variable 'a' should have a type"
    assert var_b.slang is not None, "Variable 'b' should have a type"
    assert var_result.slang is not None, "Variable 'result' should have a type"

    # ft_add takes two ints and returns an int
    assert var_a.slang.name == "int"
    assert var_b.slang.name == "int"
    assert var_result.slang.name == "int"

    print("\n" + program.dump())


def test_type_inference_sequential_calls():
    """Test type propagation through sequential function calls."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Build program: temp = mul(a, b); result = add(temp, c)
    builder = FuseProgramBuilder("test_type_propagation")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_slang(ft_add, [temp, c], [result])

    program = builder.build()

    # Run type inference with bind context and input marshals
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 2, 3, 5)
    success = program.infer_types(context, bindings.args)
    assert success, "Type inference should succeed"

    # Check that temp got its type from mul's return type
    var_temp = program.get_variable(temp)
    assert var_temp.slang is not None
    assert var_temp.slang.name == "int"

    # Check all other variables
    assert program.get_variable(a).slang.name == "int"
    assert program.get_variable(b).slang.name == "int"
    assert program.get_variable(c).slang.name == "int"
    assert program.get_variable(result).slang.name == "int"

    print("\n" + program.dump())


def test_type_inference_complex_chain():
    """Test type inference on a complex chain of operations."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")
    ft_mul = module.require_function("ft_mul")

    # Build program with multiple operations
    # t1 = add(a, b)
    # t2 = mul(c, d)
    # result = add(t1, t2)
    builder = FuseProgramBuilder("complex_chain")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    t1 = builder.temp("t1")
    t2 = builder.temp("t2")
    result = builder.output("result")

    builder.call_slang(ft_add, [a, b], [t1])
    builder.call_slang(ft_mul, [c, d], [t2])
    builder.call_slang(ft_add, [t1, t2], [result])

    program = builder.build()

    # Run type inference with bind context and input marshals
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 1, 2, 3, 4)
    success = program.infer_types(context, bindings.args)
    assert success

    # All variables should have int type
    for var in program.variables:
        assert var.slang is not None
        assert var.slang.name == "int"

    print("\n" + program.dump())


def test_type_inference_with_generics():
    """Test type inference with generic functions."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_generic_add = module.require_function("ft_generic_add")

    # Build program using generic function
    builder = FuseProgramBuilder("test_generic")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_generic_add, [a, b], [result])

    program = builder.build()

    # Run type inference with bind context and input marshals
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 5, 3)
    success = program.infer_types(context, bindings.args)

    # Note: For generic functions, type inference may be more complex
    # For now, we just check that the mechanism runs without errors
    print("\n" + program.dump())


def test_type_inference_sub_program():
    """Test type inference with sub-programs."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")
    ft_mul = module.require_function("ft_mul")

    # Create sub-program with type inference
    sub_builder = FuseProgramBuilder("add_sub")
    sub_x = sub_builder.input("x")
    sub_y = sub_builder.input("y")
    sub_result = sub_builder.output("result")
    sub_builder.call_slang(ft_add, [sub_x, sub_y], [sub_result])
    sub_program = sub_builder.build()

    # Infer types in sub-program first
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    sub_bindings = BoundCall(context, 1, 2)
    sub_program.infer_types(context, sub_bindings.args)

    # Create main program that uses the sub-program
    builder = FuseProgramBuilder("main_with_typed_sub")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_sub(sub_program, [temp, c], [result])

    program = builder.build()

    # Run type inference on main program
    bindings = BoundCall(context, 3, 4, 5)
    success = program.infer_types(context, bindings.args)
    assert success

    # Check that types were propagated from sub-program
    assert program.get_variable(result).slang is not None
    assert program.get_variable(result).slang.name == "int"

    print("\n" + program.dump())


def test_sub_program_with_intermediate_types():
    """
    Test that sub-programs work when receiving inputs from slang function outputs.
    This tests the case where intermediate variables have slang types but no python marshals.
    """
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create sub-program that takes two inputs and adds them
    sub_builder = FuseProgramBuilder("add_sub")
    sub_x = sub_builder.input("x")
    sub_y = sub_builder.input("y")
    sub_result = sub_builder.output("result")
    sub_builder.call_slang(ft_add, [sub_x, sub_y], [sub_result])
    sub_program = sub_builder.build()

    # Create main program:
    # temp1 = mul(a, b)  - temp1 will have slang type but no python marshal
    # temp2 = mul(c, d)  - temp2 will have slang type but no python marshal
    # result = sub_program(temp1, temp2)  - passes intermediate results to sub
    builder = FuseProgramBuilder("main_with_intermediate")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp1])
    builder.call_slang(ft_mul, [c, d], [temp2])
    builder.call_sub(sub_program, [temp1, temp2], [result])

    program = builder.build()

    # Run type inference - only the original inputs have marshals
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 2, 3, 4, 5)
    success = program.infer_types(context, bindings.args)
    assert success, "Type inference should succeed when passing intermediate results to sub-program"

    # Verify all variables have correct types
    assert program.get_variable(a).slang is not None
    assert program.get_variable(a).slang.name == "int"
    assert program.get_variable(temp1).slang is not None
    assert program.get_variable(temp1).slang.name == "int"
    assert program.get_variable(temp2).slang is not None
    assert program.get_variable(temp2).slang.name == "int"
    assert program.get_variable(result).slang is not None
    assert program.get_variable(result).slang.name == "int"

    # Verify sub-program was properly typed
    sub_x_var = sub_program.get_variable(sub_program.input_vars[0])
    assert sub_x_var.slang is not None
    assert sub_x_var.slang.name == "int"

    print("\n" + program.dump())


def test_clear_types():
    """Test clearing type information."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    # Build and infer types
    builder = FuseProgramBuilder("test_clear")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 10, 20)
    program.infer_types(context, bindings.args)

    # Verify types are set
    assert all(var.slang is not None for var in program.variables)

    # Clear types
    program.clear_types()

    # Verify types are cleared
    assert all(var.slang is None for var in program.variables)

    # Re-infer types (need to provide bind_context again)
    bindings2 = BoundCall(context, 15, 25)
    success = program.infer_types(context, bindings2.args)
    assert success
    assert all(var.slang is not None for var in program.variables)


# ============================================================================
# Step 5: Code Generation Tests
# ============================================================================


def test_code_generation_simple():
    """Test generating code for a simple function."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    # Build program
    builder = FuseProgramBuilder("simple_add")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 10, 20)
    program.infer_types(context, bindings.args)

    # Generate code
    code = program.generate_code()

    print("\n" + code)

    # Verify code contains expected elements
    assert "int __func_simple_add(int v0, int v1)" in code
    assert "int v2;" in code
    assert "v2 = ft_add(v0, v1);" in code
    assert "return v2;" in code


def test_code_generation_sequential():
    """Test code generation for sequential operations - matches the example."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Build program: temp = mul(a, b); result = add(temp, c)
    builder = FuseProgramBuilder("test_type_propagation")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_slang(ft_add, [temp, c], [result])

    program = builder.build()
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 2, 3, 5)
    program.infer_types(context, bindings.args)

    # Generate code
    code = program.generate_code()

    print("\n" + code)

    # Verify code matches expected format
    assert "int __func_test_type_propagation(int v0, int v1, int v2)" in code
    assert "int v3,v4;" in code or ("int v3;" in code and "int v4;" in code)
    assert "v3 = ft_mul(v0, v1);" in code
    assert "v4 = ft_add(v3, v2);" in code
    assert "return v4;" in code


def test_code_generation_complex():
    """Test code generation for complex dataflow."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")
    ft_mul = module.require_function("ft_mul")

    # Build program: t1 = add(a, b); t2 = mul(c, d); result = add(t1, t2)
    builder = FuseProgramBuilder("complex")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    t1 = builder.temp("t1")
    t2 = builder.temp("t2")
    result = builder.output("result")

    builder.call_slang(ft_add, [a, b], [t1])
    builder.call_slang(ft_mul, [c, d], [t2])
    builder.call_slang(ft_add, [t1, t2], [result])

    program = builder.build()
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 1, 2, 3, 4)
    program.infer_types(context, bindings.args)

    # Generate code
    code = program.generate_code()

    print("\n" + code)

    # Verify structure
    assert "int __func_complex(int v0, int v1, int v2, int v3)" in code
    assert "v4 = ft_add(v0, v1);" in code
    assert "v5 = ft_mul(v2, v3);" in code
    assert "v6 = ft_add(v4, v5);" in code
    assert "return v6;" in code


def test_code_generation_custom_name():
    """Test code generation with custom function name."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    builder = FuseProgramBuilder("test_prog")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 10, 20)
    program.infer_types(context, bindings.args)

    # Generate with custom name
    code = program.generate_code("my_custom_function")

    print("\n" + code)

    assert "int my_custom_function(int v0, int v1)" in code


def test_code_generation_with_sub_program():
    """Test code generation with sub-programs."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")
    ft_mul = module.require_function("ft_mul")

    # Create sub-program
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)

    sub_builder = FuseProgramBuilder("add_helper")
    sub_x = sub_builder.input("x")
    sub_y = sub_builder.input("y")
    sub_result = sub_builder.output("result")
    sub_builder.call_slang(ft_add, [sub_x, sub_y], [sub_result])
    sub_program = sub_builder.build()
    sub_bindings = BoundCall(context, 1, 2)
    sub_program.infer_types(context, sub_bindings.args)

    # Create main program
    builder = FuseProgramBuilder("main_func")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_sub(sub_program, [temp, c], [result])

    program = builder.build()
    bindings = BoundCall(context, 3, 4, 5)
    program.infer_types(context, bindings.args)

    # Generate code for both programs
    sub_code = sub_program.generate_code()
    main_code = program.generate_code()

    print("\nSub-program code:")
    print(sub_code)
    print("\nMain program code:")
    print(main_code)

    # Verify sub-program code
    assert "int __func_add_helper(int v0, int v1)" in sub_code
    assert "v2 = ft_add(v0, v1);" in sub_code

    # Verify main program calls sub-program
    assert "int __func_main_func(int v0, int v1, int v2)" in main_code
    assert "v3 = ft_mul(v0, v1);" in main_code
    assert "v4 = __func_add_helper(v3, v2);" in main_code


def test_code_generation_error_without_types():
    """Test that code generation fails if types aren't inferred."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    builder = FuseProgramBuilder("test_no_types")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()
    # Don't run type inference

    # Should raise error
    try:
        code = program.generate_code()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not all variables have types" in str(e)
        print(f"\nCorrectly caught error: {e}")


# ============================================================================
# Binding Integration Tests
# ============================================================================


def test_type_inference_with_bindings_simple():
    """Test type inference using bindings for a simple function call."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    # Build program
    builder = FuseProgramBuilder("test_with_bindings")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()

    # Create bind context
    from slangpy.bindings.marshall import BindContext
    from slangpy.bindings.boundvariable import BoundCall
    from slangpy.core.native import CallMode, CallDataMode

    context = BindContext(
        module.layout,
        CallMode.prim,
        module.device_module,
        {},
        CallDataMode.global_data,
    )

    # Create bindings with actual integer values
    bindings = BoundCall(context, 10, 20)

    # Run type inference with bindings
    success = program.infer_types(context, bindings.args)

    assert success, "Type inference should succeed"

    # Verify types were inferred correctly
    assert program.get_variable(a).slang is not None
    assert program.get_variable(b).slang is not None
    assert program.get_variable(result).slang is not None

    # Should still be int types
    assert program.get_variable(a).slang.name == "int"
    assert program.get_variable(b).slang.name == "int"
    assert program.get_variable(result).slang.name == "int"

    print("\n" + program.dump())


def test_type_inference_with_bindings_sequential():
    """Test binding propagation through sequential operations."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Build program
    builder = FuseProgramBuilder("test_bindings_sequential")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_slang(ft_add, [temp, c], [result])

    program = builder.build()

    # Create bind context
    from slangpy.bindings.marshall import BindContext
    from slangpy.bindings.boundvariable import BoundCall
    from slangpy.core.native import CallMode, CallDataMode

    context = BindContext(
        module.layout,
        CallMode.prim,
        module.device_module,
        {},
        CallDataMode.global_data,
    )

    # Create bindings
    bindings = BoundCall(context, 5, 3, 7)

    # Run type inference with bindings
    success = program.infer_types(context, bindings.args)

    assert success, "Type inference should succeed"

    # Verify all variables have types
    for var in program.variables:
        assert var.slang is not None, f"Variable {var.name} should have a type"
        assert var.slang.name == "int"

    print("\n" + program.dump())


def test_backward_compatibility_no_bindings():
    """Test that type inference without bind_context raises an error."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    builder = FuseProgramBuilder("test_no_bindings")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_add, [a, b], [result])

    program = builder.build()

    # Run type inference WITHOUT bind_context should now raise ValueError
    try:
        success = program.infer_types()
        assert False, "Expected ValueError when calling infer_types() without bind_context"
    except ValueError as e:
        assert "bind_context is required" in str(e)


def test_binding_propagation_to_outputs():
    """Test that bindings are propagated to output variables."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")

    builder = FuseProgramBuilder("test_output_binding")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_mul, [a, b], [result])

    program = builder.build()

    # Create bind context and bindings
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 4, 5)

    # Run type inference
    program.infer_types(context, bindings.args)

    # Check that input variables have marshals
    assert program.get_variable(a).python is not None
    assert program.get_variable(b).python is not None

    # Verify types
    assert program.get_variable(result).slang is not None
    assert program.get_variable(result).slang.name == "int"


def test_generic_function_resolution():
    """Test that generic functions can be properly resolved with type inference."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_generic_add = module.require_function("ft_generic_add")

    builder = FuseProgramBuilder("test_generic")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")
    builder.call_slang(ft_generic_add, [a, b], [result])

    program = builder.build()

    # Create bind context and bindings with int arguments
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)

    # Create bindings with int arguments to specialize the generic function
    bindings = BoundCall(context, 5, 3)

    # Run type inference with the bind context
    success = program.infer_types(context, bindings.args)
    assert success, "Type inference should succeed for generic function"

    # Verify types were resolved correctly
    assert program.get_variable(a).slang is not None
    assert program.get_variable(a).slang.name == "int"
    assert program.get_variable(b).slang is not None
    assert program.get_variable(b).slang.name == "int"
    assert program.get_variable(result).slang is not None
    assert program.get_variable(result).slang.name == "int"


def test_mixed_input_sources():
    """
    Test that type resolution works when a function takes inputs both directly
    from program inputs and from previous operation outputs.
    """
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    builder = FuseProgramBuilder("test_mixed")
    # Program has 3 inputs: x, y, z
    x = builder.input("x")
    y = builder.input("y")
    z = builder.input("z")

    # First operation: mul(x, y) -> temp
    temp = builder.temp("temp")
    builder.call_slang(ft_mul, [x, y], [temp])

    # Second operation: add(temp, z) -> result
    # This takes one input from previous operation (temp) and one from program input (z)
    result = builder.output("result")
    builder.call_slang(ft_add, [temp, z], [result])

    program = builder.build()

    # Create bind context with three int arguments
    from slangpy.bindings.boundvariable import BoundCall

    context = create_bind_context(module)
    bindings = BoundCall(context, 2, 3, 5)

    # Run type inference
    success = program.infer_types(context, bindings.args)
    assert success, "Type inference should succeed with mixed inputs"

    # Verify all variables have correct types
    assert program.get_variable(x).slang.name == "int"
    assert program.get_variable(y).slang.name == "int"
    assert program.get_variable(z).slang.name == "int"
    assert program.get_variable(temp).slang.name == "int"
    assert program.get_variable(result).slang.name == "int"

    # Verify we can generate code
    code = program.generate_code()
    assert "int __func_test_mixed(int v" in code or "int32_t __func_test_mixed(int32_t v" in code


def test_fused_program_callable():
    """
    Test that a fused program can be turned into a callable function
    using the Module.create_fused_function() method.
    """
    import slangpy as spy

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create a fused program: result = add(mul(a, b), c)
    builder = FuseProgramBuilder("fused_add_mul")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_slang(ft_add, [temp, c], [result])

    fuse_program = builder.build()

    # Create a callable function from the fused program
    fused_func = module.create_fused_function(fuse_program)

    # Test calling the fused function
    # This should trigger the full resolution and code generation pipeline
    result_val = fused_func(3, 4, 5)

    # Expected: (3 * 4) + 5 = 17
    assert result_val == 17, f"Expected 17, got {result_val}"
