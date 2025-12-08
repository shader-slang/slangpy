# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Comprehensive tests for fused programs.
Tests various patterns of program fusion and execution.
"""

import slangpy as spy
import slangpy.testing.helpers as helpers
from slangpy.experimental.fusevm import FuseProgramBuilder


def test_simple_sequential_fusion():
    """Test a simple sequential fusion: result = add(mul(a, b), c)"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create fused program: result = add(mul(a, b), c)
    builder = FuseProgramBuilder("simple_sequential")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_slang(ft_add, [temp, c], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (3 * 4) + 5 = 17
    result_val = fused_func(3, 4, 5)
    assert result_val == 17, f"Expected 17, got {result_val}"


def test_multiple_operations_chain():
    """Test a longer chain: result = sub(mul(add(a, b), c), d)"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")
    ft_mul = module.require_function("ft_mul")
    ft_sub = module.require_function("ft_sub")

    # Create fused program: result = sub(mul(add(a, b), c), d)
    builder = FuseProgramBuilder("chain_operations")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_add, [a, b], [temp1])  # temp1 = a + b
    builder.call_slang(ft_mul, [temp1, c], [temp2])  # temp2 = temp1 * c
    builder.call_slang(ft_sub, [temp2, d], [result])  # result = temp2 - d

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: ((2 + 3) * 4) - 1 = 19
    result_val = fused_func(2, 3, 4, 1)
    assert result_val == 19, f"Expected 19, got {result_val}"


def test_parallel_operations():
    """Test parallel operations: result = add(mul(a, b), mul(c, d))"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create fused program: result = add(mul(a, b), mul(c, d))
    builder = FuseProgramBuilder("parallel_ops")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp1])  # temp1 = a * b
    builder.call_slang(ft_mul, [c, d], [temp2])  # temp2 = c * d
    builder.call_slang(ft_add, [temp1, temp2], [result])  # result = temp1 + temp2

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (2 * 3) + (4 * 5) = 26
    result_val = fused_func(2, 3, 4, 5)
    assert result_val == 26, f"Expected 26, got {result_val}"


def test_single_operation():
    """Test a single operation fusion: result = mul(a, b)"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")

    # Create fused program: result = mul(a, b)
    builder = FuseProgramBuilder("single_op")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: 7 * 8 = 56
    result_val = fused_func(7, 8)
    assert result_val == 56, f"Expected 56, got {result_val}"


def test_unary_operation():
    """Test fusion with unary operation: result = negate(mul(a, b))"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_negate = module.require_function("ft_negate")

    # Create fused program: result = negate(mul(a, b))
    builder = FuseProgramBuilder("unary_op")
    a = builder.input("a")
    b = builder.input("b")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_slang(ft_negate, [temp], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: -(3 * 4) = -12
    result_val = fused_func(3, 4)
    assert result_val == -12, f"Expected -12, got {result_val}"


def test_complex_dag():
    """Test a more complex DAG structure"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")
    ft_mul = module.require_function("ft_mul")
    ft_sub = module.require_function("ft_sub")

    # Create fused program with DAG structure:
    # temp1 = add(a, b)
    # temp2 = mul(a, c)
    # temp3 = sub(temp1, temp2)
    # result = add(temp3, d)
    builder = FuseProgramBuilder("complex_dag")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    temp3 = builder.temp("temp3")
    result = builder.output("result")

    builder.call_slang(ft_add, [a, b], [temp1])
    builder.call_slang(ft_mul, [a, c], [temp2])
    builder.call_slang(ft_sub, [temp1, temp2], [temp3])
    builder.call_slang(ft_add, [temp3, d], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test with a=5, b=3, c=2, d=1
    # temp1 = 5 + 3 = 8
    # temp2 = 5 * 2 = 10
    # temp3 = 8 - 10 = -2
    # result = -2 + 1 = -1
    result_val = fused_func(5, 3, 2, 1)
    assert result_val == -1, f"Expected -1, got {result_val}"


def test_reusing_input_multiple_times():
    """Test fusion where an input is used multiple times"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create fused program: result = add(mul(a, a), mul(a, b))
    builder = FuseProgramBuilder("reuse_input")
    a = builder.input("a")
    b = builder.input("b")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, a], [temp1])  # temp1 = a * a
    builder.call_slang(ft_mul, [a, b], [temp2])  # temp2 = a * b
    builder.call_slang(ft_add, [temp1, temp2], [result])  # result = temp1 + temp2

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test with a=3, b=4: (3*3) + (3*4) = 9 + 12 = 21
    result_val = fused_func(3, 4)
    assert result_val == 21, f"Expected 21, got {result_val}"


def test_float_operations():
    """Test fusion with float operations"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul_float = module.require_function("ft_mul_float")
    ft_add_float = module.require_function("ft_add_float")

    # Create fused program: result = add(mul(a, b), c)
    builder = FuseProgramBuilder("float_ops")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul_float, [a, b], [temp])
    builder.call_slang(ft_add_float, [temp, c], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (2.5 * 4.0) + 1.5 = 11.5
    result_val = fused_func(2.5, 4.0, 1.5)
    assert abs(result_val - 11.5) < 0.001, f"Expected 11.5, got {result_val}"


def test_multiple_calls_same_function():
    """Test fusion with multiple calls to the same function"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    # Create fused program: result = add(add(add(a, b), c), d)
    builder = FuseProgramBuilder("repeated_add")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_add, [a, b], [temp1])
    builder.call_slang(ft_add, [temp1, c], [temp2])
    builder.call_slang(ft_add, [temp2, d], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: 1 + 2 + 3 + 4 = 10
    result_val = fused_func(1, 2, 3, 4)
    assert result_val == 10, f"Expected 10, got {result_val}"


def test_max_min_operations():
    """Test fusion with max and min operations"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_max = module.require_function("ft_max")
    ft_min = module.require_function("ft_min")
    ft_add = module.require_function("ft_add")

    # Create fused program: result = add(max(a, b), min(c, d))
    builder = FuseProgramBuilder("max_min")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_max, [a, b], [temp1])
    builder.call_slang(ft_min, [c, d], [temp2])
    builder.call_slang(ft_add, [temp1, temp2], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: max(3, 7) + min(5, 2) = 7 + 2 = 9
    result_val = fused_func(3, 7, 5, 2)
    assert result_val == 9, f"Expected 9, got {result_val}"


def test_division_operation():
    """Test fusion with division"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_div = module.require_function("ft_div")

    # Create fused program: result = div(mul(a, b), c)
    builder = FuseProgramBuilder("div_ops")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_slang(ft_div, [temp, c], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (20 * 3) / 4 = 60 / 4 = 15
    result_val = fused_func(20, 3, 4)
    assert result_val == 15, f"Expected 15, got {result_val}"


def test_generic_function_fusion():
    """Test fusion with generic functions"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_generic_add = module.require_function("ft_generic_add")
    ft_generic_mul = module.require_function("ft_generic_mul")

    # Create fused program: result = generic_add(generic_mul(a, b), c)
    builder = FuseProgramBuilder("generic_fusion")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_generic_mul, [a, b], [temp])
    builder.call_slang(ft_generic_add, [temp, c], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (5 * 6) + 7 = 37
    result_val = fused_func(5, 6, 7)
    assert result_val == 37, f"Expected 37, got {result_val}"


def test_wide_fusion():
    """Test fusion with many parallel operations"""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create fused program with wide parallelism:
    # temp1 = mul(a, b)
    # temp2 = mul(c, d)
    # temp3 = mul(e, f)
    # temp4 = add(temp1, temp2)
    # result = add(temp4, temp3)
    builder = FuseProgramBuilder("wide_fusion")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    e = builder.input("e")
    f = builder.input("f")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    temp3 = builder.temp("temp3")
    temp4 = builder.temp("temp4")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp1])
    builder.call_slang(ft_mul, [c, d], [temp2])
    builder.call_slang(ft_mul, [e, f], [temp3])
    builder.call_slang(ft_add, [temp1, temp2], [temp4])
    builder.call_slang(ft_add, [temp4, temp3], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (2*3) + (4*5) + (6*7) = 6 + 20 + 42 = 68
    result_val = fused_func(2, 3, 4, 5, 6, 7)
    assert result_val == 68, f"Expected 68, got {result_val}"


def test_vector_to_array_sum():
    """
    Test fusion with vector operations:
    - Scale a float3 vector
    - Convert to array
    - Sum the array elements
    """
    from slangpy import float3

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_scale_vec = module.require_function("ft_scale_vec_generic")
    ft_vec_to_array = module.require_function("ft_vec_to_array")
    ft_array_sum = module.require_function("ft_array_sum")

    # Create fused program:
    # scaled_vec = ft_scale_vec_generic(vec, scalar)
    # arr = ft_vec_to_array(scaled_vec)
    # result = ft_array_sum(arr)
    builder = FuseProgramBuilder("vector_to_array_sum")
    vec = builder.input("vec")
    scalar = builder.input("scalar")
    scaled_vec = builder.temp("scaled_vec")
    arr = builder.temp("arr")
    result = builder.output("result")

    builder.call_slang(ft_scale_vec, [vec, scalar], [scaled_vec])
    builder.call_slang(ft_vec_to_array, [scaled_vec], [arr])
    builder.call_slang(ft_array_sum, [arr], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: scale float3(1.0, 2.0, 3.0) by 2.0, then sum
    # scaled_vec = (2.0, 4.0, 6.0)
    # sum = 2.0 + 4.0 + 6.0 = 12.0
    vec_input = float3(1.0, 2.0, 3.0)
    result_val = fused_func(vec_input, 2.0)
    assert abs(result_val - 12.0) < 0.001, f"Expected 12.0, got {result_val}"


def test_simple_sub_program():
    """
    Test a simple sub-program:
    - Sub-program: add(x, y)
    - Main program: mul(a, b) then call sub-program with result and c
    - Expected: (a * b) + c
    """
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create sub-program: result = add(x, y)
    sub_builder = FuseProgramBuilder("add_sub")
    sub_x = sub_builder.input("x")
    sub_y = sub_builder.input("y")
    sub_result = sub_builder.output("result")
    sub_builder.call_slang(ft_add, [sub_x, sub_y], [sub_result])
    sub_program = sub_builder.build()

    # Create main program that uses the sub-program
    builder = FuseProgramBuilder("main_with_sub")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_mul, [a, b], [temp])
    builder.call_sub(sub_program, [temp, c], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (3 * 4) + 5 = 17
    result_val = fused_func(3, 4, 5)
    assert result_val == 17, f"Expected 17, got {result_val}"


def test_nested_sub_programs():
    """
    Test nested sub-programs:
    - Inner sub-program: mul(x, y)
    - Outer sub-program: calls inner with (a, b), then adds c
    - Main program: calls outer with inputs
    - Expected: (a * b) + c
    """
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create inner sub-program: result = mul(x, y)
    inner_builder = FuseProgramBuilder("mul_inner")
    inner_x = inner_builder.input("x")
    inner_y = inner_builder.input("y")
    inner_result = inner_builder.output("result")
    inner_builder.call_slang(ft_mul, [inner_x, inner_y], [inner_result])
    inner_program = inner_builder.build()

    # Create outer sub-program: temp = mul_inner(a, b), result = add(temp, c)
    outer_builder = FuseProgramBuilder("mul_add_outer")
    outer_a = outer_builder.input("a")
    outer_b = outer_builder.input("b")
    outer_c = outer_builder.input("c")
    outer_temp = outer_builder.temp("temp")
    outer_result = outer_builder.output("result")
    outer_builder.call_sub(inner_program, [outer_a, outer_b], [outer_temp])
    outer_builder.call_slang(ft_add, [outer_temp, outer_c], [outer_result])
    outer_program = outer_builder.build()

    # Create main program that calls the outer sub-program
    builder = FuseProgramBuilder("main_nested")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    result = builder.output("result")
    builder.call_sub(outer_program, [a, b, c], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (3 * 4) + 5 = 17
    result_val = fused_func(3, 4, 5)
    assert result_val == 17, f"Expected 17, got {result_val}"


def test_multiple_sub_programs():
    """
    Test using multiple different sub-programs:
    - Sub-program 1: mul(x, y)
    - Sub-program 2: add(x, y)
    - Main: temp1 = mul_sub(a, b), temp2 = mul_sub(c, d), result = add_sub(temp1, temp2)
    - Expected: (a * b) + (c * d)
    """
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create mul sub-program
    mul_sub_builder = FuseProgramBuilder("mul_sub")
    mul_x = mul_sub_builder.input("x")
    mul_y = mul_sub_builder.input("y")
    mul_result = mul_sub_builder.output("result")
    mul_sub_builder.call_slang(ft_mul, [mul_x, mul_y], [mul_result])
    mul_sub_program = mul_sub_builder.build()

    # Create add sub-program
    add_sub_builder = FuseProgramBuilder("add_sub")
    add_x = add_sub_builder.input("x")
    add_y = add_sub_builder.input("y")
    add_result = add_sub_builder.output("result")
    add_sub_builder.call_slang(ft_add, [add_x, add_y], [add_result])
    add_sub_program = add_sub_builder.build()

    # Create main program using both sub-programs
    builder = FuseProgramBuilder("main_multiple_subs")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_sub(mul_sub_program, [a, b], [temp1])
    builder.call_sub(mul_sub_program, [c, d], [temp2])
    builder.call_sub(add_sub_program, [temp1, temp2], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (3 * 4) + (5 * 2) = 12 + 10 = 22
    result_val = fused_func(3, 4, 5, 2)
    assert result_val == 22, f"Expected 22, got {result_val}"


def test_sub_program_with_chain():
    """
    Test sub-program containing a chain of operations:
    - Sub-program: temp = mul(a, b), result = add(temp, c)
    - Main: result = sub(x, y, z)
    - Expected: (x * y) + z
    """
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul = module.require_function("ft_mul")
    ft_add = module.require_function("ft_add")

    # Create sub-program with chain: (a * b) + c
    sub_builder = FuseProgramBuilder("mul_add_chain")
    sub_a = sub_builder.input("a")
    sub_b = sub_builder.input("b")
    sub_c = sub_builder.input("c")
    sub_temp = sub_builder.temp("temp")
    sub_result = sub_builder.output("result")
    sub_builder.call_slang(ft_mul, [sub_a, sub_b], [sub_temp])
    sub_builder.call_slang(ft_add, [sub_temp, sub_c], [sub_result])
    sub_program = sub_builder.build()

    # Create main program that calls the sub-program
    builder = FuseProgramBuilder("main_chain_sub")
    x = builder.input("x")
    y = builder.input("y")
    z = builder.input("z")
    result = builder.output("result")
    builder.call_sub(sub_program, [x, y, z], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (3 * 4) + 5 = 17
    result_val = fused_func(3, 4, 5)
    assert result_val == 17, f"Expected 17, got {result_val}"


def test_sub_program_reuse():
    """
    Test reusing the same sub-program multiple times:
    - Sub-program: add(x, y)
    - Main: temp1 = add_sub(a, b), temp2 = add_sub(c, d), result = add_sub(temp1, temp2)
    - Expected: (a + b) + (c + d)
    """
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_add = module.require_function("ft_add")

    # Create add sub-program
    sub_builder = FuseProgramBuilder("add_sub")
    sub_x = sub_builder.input("x")
    sub_y = sub_builder.input("y")
    sub_result = sub_builder.output("result")
    sub_builder.call_slang(ft_add, [sub_x, sub_y], [sub_result])
    sub_program = sub_builder.build()

    # Create main program that reuses the sub-program three times
    builder = FuseProgramBuilder("main_reuse_sub")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_sub(sub_program, [a, b], [temp1])
    builder.call_sub(sub_program, [c, d], [temp2])
    builder.call_sub(sub_program, [temp1, temp2], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Test: (1 + 2) + (3 + 4) = 3 + 7 = 10
    result_val = fused_func(1, 2, 3, 4)
    assert result_val == 10, f"Expected 10, got {result_val}"


# NOTE: Multiple outputs from sub-programs are not currently supported because
# Slang functions can only return a single value. If this feature is needed in
# the future, it would require returning a struct with multiple fields.


# ============================================================================
# Tensor Vectorization Tests
# ============================================================================


def test_tensor_fusion_element_wise_operations():
    """
    Test fusion with tensor element-wise operations:
    - Create two random float tensors
    - Fuse: mul, add, mul sequence
    - Expected: ((a * b) + c) * d
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul_float = module.require_function("ft_mul_float")
    ft_add_float = module.require_function("ft_add_float")

    # Create fused program: result = mul(add(mul(a, b), c), d)
    builder = FuseProgramBuilder("tensor_element_wise")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(ft_mul_float, [a, b], [temp1])
    builder.call_slang(ft_add_float, [temp1, c], [temp2])
    builder.call_slang(ft_mul_float, [temp2, d], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(42)
    shape = (128, 256)
    a_data = np.random.randn(*shape).astype(np.float32)
    b_data = np.random.randn(*shape).astype(np.float32)
    c_data = np.random.randn(*shape).astype(np.float32)
    d_data = np.random.randn(*shape).astype(np.float32)

    # Create tensors
    tensor_a = Tensor.from_numpy(device, a_data)
    tensor_b = Tensor.from_numpy(device, b_data)
    tensor_c = Tensor.from_numpy(device, c_data)
    tensor_d = Tensor.from_numpy(device, d_data)

    # Call fused function with tensors
    result_tensor = fused_func(tensor_a, tensor_b, tensor_c, tensor_d, _result="tensor")

    # Get numpy array back
    result_data = result_tensor.to_numpy()

    # Verify results: ((a * b) + c) * d
    expected = ((a_data * b_data) + c_data) * d_data
    assert result_data.shape == expected.shape
    assert np.allclose(result_data, expected, rtol=1e-5, atol=1e-6)


def test_tensor_fusion_with_sub_program():
    """
    Test fusion with sub-program on tensors:
    - Sub-program: square_diff = (a - b) * (a - b)
    - Main: sqrt of square_diff
    - Expected: sqrt((a - b)^2) = |a - b|
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_sub_float = module.require_function("ft_sub_float")
    ft_mul_float = module.require_function("ft_mul_float")
    ft_sqrt_float = module.require_function("ft_sqrt_float")

    # Create sub-program: square_diff = (a - b) * (a - b)
    sub_builder = FuseProgramBuilder("square_diff_sub")
    sub_a = sub_builder.input("a")
    sub_b = sub_builder.input("b")
    sub_diff = sub_builder.temp("diff")
    sub_result = sub_builder.output("result")
    sub_builder.call_slang(ft_sub_float, [sub_a, sub_b], [sub_diff])
    sub_builder.call_slang(ft_mul_float, [sub_diff, sub_diff], [sub_result])
    sub_program = sub_builder.build()

    # Create main program: sqrt of sub-program result
    builder = FuseProgramBuilder("tensor_distance")
    a = builder.input("a")
    b = builder.input("b")
    square_diff = builder.temp("square_diff")
    result = builder.output("result")

    builder.call_sub(sub_program, [a, b], [square_diff])
    builder.call_slang(ft_sqrt_float, [square_diff], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(123)
    shape = (64, 128)
    a_data = np.random.randn(*shape).astype(np.float32) * 10 + 5
    b_data = np.random.randn(*shape).astype(np.float32) * 10 + 5

    # Create tensors
    tensor_a = Tensor.from_numpy(device, a_data)
    tensor_b = Tensor.from_numpy(device, b_data)

    # Call fused function with tensors
    result_tensor = fused_func(tensor_a, tensor_b, _result="tensor")

    # Get numpy array back
    result_data = result_tensor.to_numpy()

    # Verify results: sqrt((a - b)^2) = |a - b|
    expected = np.abs(a_data - b_data)
    assert result_data.shape == expected.shape
    assert np.allclose(result_data, expected, rtol=1e-5, atol=1e-6)


def test_tensor_fusion_complex_chain():
    """
    Test complex fusion chain with tensors:
    - Multiple operations in parallel and sequential
    - Sub-program: weighted_sum = (a * w1) + (b * w2)
    - Main: applies sub twice, then adds results
    - Expected: ((a * w1) + (b * w2)) + ((c * w3) + (d * w4))
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_mul_float = module.require_function("ft_mul_float")
    ft_add_float = module.require_function("ft_add_float")

    # Create sub-program: weighted_sum = (x * w1) + (y * w2)
    sub_builder = FuseProgramBuilder("weighted_sum")
    sub_x = sub_builder.input("x")
    sub_y = sub_builder.input("y")
    sub_w1 = sub_builder.input("w1")
    sub_w2 = sub_builder.input("w2")
    sub_prod1 = sub_builder.temp("prod1")
    sub_prod2 = sub_builder.temp("prod2")
    sub_result = sub_builder.output("result")

    sub_builder.call_slang(ft_mul_float, [sub_x, sub_w1], [sub_prod1])
    sub_builder.call_slang(ft_mul_float, [sub_y, sub_w2], [sub_prod2])
    sub_builder.call_slang(ft_add_float, [sub_prod1, sub_prod2], [sub_result])
    sub_program = sub_builder.build()

    # Create main program: use sub-program twice and combine
    builder = FuseProgramBuilder("tensor_complex_fusion")
    a = builder.input("a")
    b = builder.input("b")
    c = builder.input("c")
    d = builder.input("d")
    w1 = builder.input("w1")
    w2 = builder.input("w2")
    w3 = builder.input("w3")
    w4 = builder.input("w4")
    sum1 = builder.temp("sum1")
    sum2 = builder.temp("sum2")
    result = builder.output("result")

    builder.call_sub(sub_program, [a, b, w1, w2], [sum1])
    builder.call_sub(sub_program, [c, d, w3, w4], [sum2])
    builder.call_slang(ft_add_float, [sum1, sum2], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(456)
    shape = (32, 64)
    a_data = np.random.randn(*shape).astype(np.float32)
    b_data = np.random.randn(*shape).astype(np.float32)
    c_data = np.random.randn(*shape).astype(np.float32)
    d_data = np.random.randn(*shape).astype(np.float32)
    w1_data = np.random.randn(*shape).astype(np.float32)
    w2_data = np.random.randn(*shape).astype(np.float32)
    w3_data = np.random.randn(*shape).astype(np.float32)
    w4_data = np.random.randn(*shape).astype(np.float32)

    # Create tensors
    tensor_a = Tensor.from_numpy(device, a_data)
    tensor_b = Tensor.from_numpy(device, b_data)
    tensor_c = Tensor.from_numpy(device, c_data)
    tensor_d = Tensor.from_numpy(device, d_data)
    tensor_w1 = Tensor.from_numpy(device, w1_data)
    tensor_w2 = Tensor.from_numpy(device, w2_data)
    tensor_w3 = Tensor.from_numpy(device, w3_data)
    tensor_w4 = Tensor.from_numpy(device, w4_data)

    # Call fused function with tensors
    result_tensor = fused_func(
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_d,
        tensor_w1,
        tensor_w2,
        tensor_w3,
        tensor_w4,
        _result="tensor",
    )

    # Get numpy array back
    result_data = result_tensor.to_numpy()

    # Verify results: ((a * w1) + (b * w2)) + ((c * w3) + (d * w4))
    expected = ((a_data * w1_data) + (b_data * w2_data)) + ((c_data * w3_data) + (d_data * w4_data))
    assert result_data.shape == expected.shape
    assert np.allclose(result_data, expected, rtol=1e-5, atol=1e-6)


# ============================================================================
# Struct-based Tests
# ============================================================================


def test_struct():
    """
    Test fusion with struct tensor and scalar:
    - Tensor of FloatContainer structs
    - Scalar FloatContainer dict
    - Expected: a.value + b.value
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_struct_add = module.require_function("ft_struct_add")

    # Create fused program: result = a.value + b.value
    builder = FuseProgramBuilder("struct_add")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")

    builder.call_slang(ft_struct_add, [a, b], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(42)
    shape = (128, 256)
    a_data = np.random.randn(*shape).astype(np.float32)

    # Create a tensor to contain a, and a dictionary to store a single struct for b
    tensor_a = Tensor.empty(device, shape=shape, dtype=module.FloatContainer)
    tensor_a.copy_from_numpy(a_data)
    value_b = {"value": 5.0}

    # Call fused function with tensors
    result_tensor = fused_func(tensor_a, value_b, _result="tensor")

    # Get numpy array back
    result_data = result_tensor.to_numpy()

    # Verify results: a.value+b.value
    expected = a_data + 5.0
    assert result_data.shape == expected.shape
    assert np.allclose(result_data, expected, rtol=1e-5, atol=1e-6)


def test_struct_two_tensors():
    """
    Test fusion with two struct tensors:
    - Two tensors of FloatContainer structs
    - Expected: a.value + b.value
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_struct_add = module.require_function("ft_struct_add")

    # Create fused program: result = a.value + b.value
    builder = FuseProgramBuilder("struct_add_two_tensors")
    a = builder.input("a")
    b = builder.input("b")
    result = builder.output("result")

    builder.call_slang(ft_struct_add, [a, b], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(123)
    shape = (64, 128)
    a_data = np.random.randn(*shape).astype(np.float32)
    b_data = np.random.randn(*shape).astype(np.float32)

    # Create tensors of FloatContainer structs
    tensor_a = Tensor.empty(device, shape=shape, dtype=module.FloatContainer)
    tensor_a.copy_from_numpy(a_data)

    tensor_b = Tensor.empty(device, shape=shape, dtype=module.FloatContainer)
    tensor_b.copy_from_numpy(b_data)

    # Call fused function with tensors
    result_tensor = fused_func(tensor_a, tensor_b, _result="tensor")

    # Get numpy array back
    result_data = result_tensor.to_numpy()

    # Verify results: a.value + b.value
    expected = a_data + b_data
    assert result_data.shape == expected.shape
    assert np.allclose(result_data, expected, rtol=1e-5, atol=1e-6)


def test_struct_mul_scalar():
    """
    Test fusion with struct multiplication:
    - Tensor of FloatContainer structs
    - Multiply by scalar
    - Extract value
    - Expected: (a.value * scalar)
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_struct_mul_scalar = module.require_function("ft_struct_mul_scalar")
    ft_struct_get_value = module.require_function("ft_struct_get_value")

    # Create fused program: result = get_value(mul_scalar(a, scalar))
    builder = FuseProgramBuilder("struct_mul_get")
    a = builder.input("a")
    scalar = builder.input("scalar")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(ft_struct_mul_scalar, [a, scalar], [temp])
    builder.call_slang(ft_struct_get_value, [temp], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(456)
    shape = (32, 64)
    a_data = np.random.randn(*shape).astype(np.float32)

    # Create tensor of FloatContainer structs
    tensor_a = Tensor.empty(device, shape=shape, dtype=module.FloatContainer)
    tensor_a.copy_from_numpy(a_data)

    scalar_value = 3.5

    # Call fused function
    result_tensor = fused_func(tensor_a, scalar_value, _result="tensor")

    # Get numpy array back
    result_data = result_tensor.to_numpy()

    # Verify results: a.value * scalar
    expected = a_data * scalar_value
    assert result_data.shape == expected.shape
    assert np.allclose(result_data, expected, rtol=1e-5, atol=1e-6)


def test_struct_with_sub_program():
    """
    Test fusion with struct and sub-program:
    - Sub-program: mul_scalar(struct, scalar)
    - Main: add two mul_scalar results
    - Expected: (a.value * s1) + (b.value * s2)
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_struct_mul_scalar = module.require_function("ft_struct_mul_scalar")
    ft_struct_add = module.require_function("ft_struct_add")

    # Create sub-program: mul_scalar
    sub_builder = FuseProgramBuilder("struct_mul_sub")
    sub_struct = sub_builder.input("struct")
    sub_scalar = sub_builder.input("scalar")
    sub_result = sub_builder.output("result")
    sub_builder.call_slang(ft_struct_mul_scalar, [sub_struct, sub_scalar], [sub_result])
    sub_program = sub_builder.build()

    # Create main program: add(mul_scalar(a, s1), mul_scalar(b, s2))
    # Note: ft_struct_add returns float, not FloatContainer
    builder = FuseProgramBuilder("struct_add_scaled")
    a = builder.input("a")
    s1 = builder.input("s1")
    b = builder.input("b")
    s2 = builder.input("s2")
    scaled_a = builder.temp("scaled_a")
    scaled_b = builder.temp("scaled_b")
    result = builder.output("result")

    builder.call_sub(sub_program, [a, s1], [scaled_a])
    builder.call_sub(sub_program, [b, s2], [scaled_b])
    builder.call_slang(ft_struct_add, [scaled_a, scaled_b], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(789)
    shape = (16, 32)
    a_data = np.random.randn(*shape).astype(np.float32)
    b_data = np.random.randn(*shape).astype(np.float32)

    # Create tensors of FloatContainer structs
    tensor_a = Tensor.empty(device, shape=shape, dtype=module.FloatContainer)
    tensor_a.copy_from_numpy(a_data)

    tensor_b = Tensor.empty(device, shape=shape, dtype=module.FloatContainer)
    tensor_b.copy_from_numpy(b_data)

    s1_value = 2.0
    s2_value = 3.0

    # Call fused function
    result_tensor = fused_func(tensor_a, s1_value, tensor_b, s2_value, _result="tensor")

    # Get numpy array back
    result_data = result_tensor.to_numpy()

    # Verify results: (a.value * s1) + (b.value * s2)
    expected = (a_data * s1_value) + (b_data * s2_value)
    assert result_data.shape == expected.shape
    assert np.allclose(result_data, expected, rtol=1e-5, atol=1e-6)


def test_struct_chain_operations():
    """
    Test fusion with chain of struct operations:
    - Chain: add(struct_a, struct_b) -> mul_scalar -> get_value
    - Expected: ((a.value + b.value) * scalar)
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    ft_struct_add = module.require_function("ft_struct_add")
    ft_mul_float = module.require_function("ft_mul_float")

    # Create fused program: result = mul(add(a, b), scalar)
    # Note: ft_struct_add returns a float, so we use ft_mul_float
    builder = FuseProgramBuilder("struct_chain")
    a = builder.input("a")
    b = builder.input("b")
    scalar = builder.input("scalar")
    sum_val = builder.temp("sum_val")
    result = builder.output("result")

    builder.call_slang(ft_struct_add, [a, b], [sum_val])
    builder.call_slang(ft_mul_float, [sum_val, scalar], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(101)
    shape = (48, 96)
    a_data = np.random.randn(*shape).astype(np.float32)
    b_data = np.random.randn(*shape).astype(np.float32)

    # Create tensors of FloatContainer structs
    tensor_a = Tensor.empty(device, shape=shape, dtype=module.FloatContainer)
    tensor_a.copy_from_numpy(a_data)

    tensor_b = Tensor.empty(device, shape=shape, dtype=module.FloatContainer)
    tensor_b.copy_from_numpy(b_data)

    scalar_value = 1.5

    # Call fused function
    result_tensor = fused_func(tensor_a, tensor_b, scalar_value, _result="tensor")

    # Get numpy array back
    result_data = result_tensor.to_numpy()

    # Verify results: (a.value + b.value) * scalar
    expected = (a_data + b_data) * scalar_value
    assert result_data.shape == expected.shape
    assert np.allclose(result_data, expected, rtol=1e-5, atol=1e-6)


# ============================================================================
# Method Call Tests (static and non-static)
# ============================================================================


def test_non_static_method_call():
    """
    Test fusion with non-static method call.
    - Input: TypeWithMethod struct, float scalar
    - Operation: struct.multiply_by_scalar(scalar)
    - Expected: struct.value * scalar
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Get the struct type and its method
    TypeWithMethod = module.require_struct("TypeWithMethod")
    multiply_method = module.find_function_in_struct(TypeWithMethod, "multiply_by_scalar")

    # Create fused program: result = struct.multiply_by_scalar(scalar)
    builder = FuseProgramBuilder("non_static_method")
    struct_input = builder.input("struct_input")
    scalar_input = builder.input("scalar")
    result = builder.output("result")

    builder.call_slang(multiply_method, [struct_input, scalar_input], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(100)
    shape = (16, 32)
    struct_data = np.random.randn(*shape).astype(np.float32)

    # Create tensor of TypeWithMethod structs
    tensor_struct = Tensor.empty(device, shape=shape, dtype=TypeWithMethod)
    tensor_struct.copy_from_numpy(struct_data)

    scalar_value = 2.5

    # Call fused function
    result_tensor = fused_func(tensor_struct, scalar_value, _result="tensor")

    # Verify results
    result_cpu = result_tensor.to_numpy()
    expected = struct_data * scalar_value
    assert np.allclose(result_cpu, expected, rtol=1e-5), f"Expected {expected}, got {result_cpu}"


def test_static_method_call():
    """
    Test fusion with static method call.
    - Input: TypeWithMethod struct, float scalar
    - Operation: TypeWithMethod.static_multiply_by_scalar(struct, scalar)
    - Expected: struct.value * scalar
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Get the struct type and its static method
    TypeWithMethod = module.require_struct("TypeWithMethod")
    static_method = module.find_function_in_struct(TypeWithMethod, "static_multiply_by_scalar")

    # Create fused program: result = TypeWithMethod.static_multiply_by_scalar(struct, scalar)
    builder = FuseProgramBuilder("static_method")
    struct_input = builder.input("struct_input")
    scalar_input = builder.input("scalar")
    result = builder.output("result")

    builder.call_slang(static_method, [struct_input, scalar_input], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(101)
    shape = (16, 32)
    struct_data = np.random.randn(*shape).astype(np.float32)

    # Create tensor of TypeWithMethod structs
    tensor_struct = Tensor.empty(device, shape=shape, dtype=TypeWithMethod)
    tensor_struct.copy_from_numpy(struct_data)

    scalar_value = 3.0

    # Call fused function
    result_tensor = fused_func(tensor_struct, scalar_value, _result="tensor")

    # Verify results
    result_cpu = result_tensor.to_numpy()
    expected = struct_data * scalar_value
    assert np.allclose(result_cpu, expected, rtol=1e-5), f"Expected {expected}, got {result_cpu}"


def test_method_call_in_chain():
    """
    Test fusion with method call in a chain.
    - Operations: struct.multiply_by_scalar(s1) -> mul_float(result, s2)
    - Expected: (struct.value * s1) * s2
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Get the struct type and its method
    TypeWithMethod = module.require_struct("TypeWithMethod")
    multiply_method = module.find_function_in_struct(TypeWithMethod, "multiply_by_scalar")
    ft_mul_float = module.require_function("ft_mul_float")

    # Create fused program: temp = struct.multiply_by_scalar(s1); result = mul_float(temp, s2)
    builder = FuseProgramBuilder("method_chain")
    struct_input = builder.input("struct_input")
    s1 = builder.input("s1")
    s2 = builder.input("s2")
    temp = builder.temp("temp")
    result = builder.output("result")

    builder.call_slang(multiply_method, [struct_input, s1], [temp])
    builder.call_slang(ft_mul_float, [temp, s2], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(102)
    shape = (24, 48)
    struct_data = np.random.randn(*shape).astype(np.float32)

    # Create tensor of TypeWithMethod structs
    tensor_struct = Tensor.empty(device, shape=shape, dtype=TypeWithMethod)
    tensor_struct.copy_from_numpy(struct_data)

    s1_value = 2.0
    s2_value = 1.5

    # Call fused function
    result_tensor = fused_func(tensor_struct, s1_value, s2_value, _result="tensor")

    # Verify results
    result_cpu = result_tensor.to_numpy()
    expected = (struct_data * s1_value) * s2_value
    assert np.allclose(result_cpu, expected, rtol=1e-5), f"Expected {expected}, got {result_cpu}"


def test_mixed_static_and_non_static():
    """
    Test fusion with both static and non-static method calls.
    - Operations: non_static_result = struct1.multiply_by_scalar(s1)
    -             static_result = TypeWithMethod.static_multiply_by_scalar(struct2, s2)
    -             result = add_float(non_static_result, static_result)
    - Expected: (struct1.value * s1) + (struct2.value * s2)
    """
    import numpy as np
    from slangpy.types import Tensor

    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Get the struct type and its methods
    TypeWithMethod = module.require_struct("TypeWithMethod")
    multiply_method = module.find_function_in_struct(TypeWithMethod, "multiply_by_scalar")
    static_method = module.find_function_in_struct(TypeWithMethod, "static_multiply_by_scalar")
    ft_add_float = module.require_function("ft_add_float")

    # Create fused program
    builder = FuseProgramBuilder("mixed_methods")
    struct1 = builder.input("struct1")
    s1 = builder.input("s1")
    struct2 = builder.input("struct2")
    s2 = builder.input("s2")
    temp1 = builder.temp("temp1")
    temp2 = builder.temp("temp2")
    result = builder.output("result")

    builder.call_slang(multiply_method, [struct1, s1], [temp1])
    builder.call_slang(static_method, [struct2, s2], [temp2])
    builder.call_slang(ft_add_float, [temp1, temp2], [result])

    fuse_program = builder.build()
    fused_func = module.create_fused_function(fuse_program)

    # Create random tensor data
    np.random.seed(103)
    shape = (8, 16)
    struct1_data = np.random.randn(*shape).astype(np.float32)
    struct2_data = np.random.randn(*shape).astype(np.float32)

    # Create tensors of TypeWithMethod structs
    tensor_struct1 = Tensor.empty(device, shape=shape, dtype=TypeWithMethod)
    tensor_struct1.copy_from_numpy(struct1_data)

    tensor_struct2 = Tensor.empty(device, shape=shape, dtype=TypeWithMethod)
    tensor_struct2.copy_from_numpy(struct2_data)

    s1_value = 2.0
    s2_value = 3.0

    # Call fused function
    result_tensor = fused_func(tensor_struct1, s1_value, tensor_struct2, s2_value, _result="tensor")

    # Verify results
    result_cpu = result_tensor.to_numpy()
    expected = (struct1_data * s1_value) + (struct2_data * s2_value)
    assert np.allclose(result_cpu, expected, rtol=1e-5), f"Expected {expected}, got {result_cpu}"
