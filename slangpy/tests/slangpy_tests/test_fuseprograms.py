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
