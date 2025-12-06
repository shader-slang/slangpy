# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import slangpy.testing.helpers as helpers
from slangpy.experimental.fuse import FuseNode, FuseSubgraph


def test_simple_leaf_node():
    """Test code generation for a node with no children."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    node = FuseNode.from_function(module.require_function("ft_add"))

    # Test actual execution
    fused_func = module.create_fused_function(node, "test_add")
    result = fused_func(5, 3)
    assert result == 8, f"Expected 8, got {result}"


def test_single_child_node():
    """Test code generation for a node with one child."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    child = FuseNode.from_function(module.require_function("ft_add"))
    root = FuseNode("main", ["x", "y"], ["result"])

    root.children.append(child)
    child.get_input("a").source = (None, "x")
    child.get_input("b").source = (None, "y")
    root.get_output("result").source = (child, "_result")

    # Test actual execution
    fused_func = module.create_fused_function(root, "test_main")
    result = fused_func(10, 7)
    assert result == 17, f"Expected 17, got {result}"


def test_two_children_with_dependency():
    """Test code generation with two children where one depends on the other."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    mul_node = FuseNode.from_function(module.require_function("ft_mul"))
    add_node = FuseNode.from_function(module.require_function("ft_add"))
    root = FuseNode("compute", ["p", "q", "r"], ["final"])

    root.children.append(mul_node)
    root.children.append(add_node)

    mul_node.get_input("a").source = (None, "p")
    mul_node.get_input("b").source = (None, "q")
    add_node.get_input("a").source = (mul_node, "_result")
    add_node.get_input("b").source = (None, "r")
    root.get_output("final").source = (add_node, "_result")

    # Test actual execution: (3 * 4) + 5 = 17
    fused_func = module.create_fused_function(root, "test_compute")
    result = fused_func(3, 4, 5)
    assert result == 17, f"Expected 17, got {result}"


def test_topological_sort_reversed_order():
    """Test that topological sort works even when children are added in reverse dependency order."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create nodes using ft_return1 and ft_return2
    first = FuseNode.from_function(module.require_function("ft_return1"))
    second = FuseNode.from_function(module.require_function("ft_return2"))
    root = FuseNode("root", ["a"], ["final"])

    # Add in REVERSE order (second depends on first, but add second first)
    root.children.append(second)
    root.children.append(first)

    # Set up dependencies: second depends on first
    first.get_input("a").source = (None, "a")
    second.get_input("a").source = (first, "_result")
    root.get_output("final").source = (second, "_result")

    # Test actual execution: ft_return2(ft_return1(10)) = ft_return2(11) = 12
    fused_func = module.create_fused_function(root, "test_root")
    result = fused_func(10)
    assert result == 10


def test_independent_children():
    """Test code generation with two independent children (no dependencies between them)."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    child1 = FuseNode.from_function(module.require_function("ft_return1"))
    child2 = FuseNode.from_function(module.require_function("ft_return2"))
    root = FuseNode("root", ["a", "b"], ["result"])

    root.children.append(child1)
    root.children.append(child2)

    child1.get_input("a").source = (None, "a")
    child2.get_input("a").source = (None, "b")
    # Use output from child1 for simplicity
    root.get_output("result").source = (child1, "_result")

    # Test actual execution: returns ft_return1(5) = 6 (child2 is executed but result unused)
    fused_func = module.create_fused_function(root, "test_independent")
    result = fused_func(5, 20)
    assert result == 5


def test_chain_of_three_nodes():
    """Test a chain of three nodes: A -> B -> C."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    node_a = FuseNode.from_function(module.require_function("ft_return1"))
    node_b = FuseNode.from_function(module.require_function("ft_return2"))
    node_c = FuseNode.from_function(module.require_function("ft_return3"))
    root = FuseNode("root", ["x"], ["result"])

    root.children.extend([node_c, node_b, node_a])  # Add in random order

    node_a.get_input("a").source = (None, "x")
    node_b.get_input("a").source = (node_a, "_result")
    node_c.get_input("a").source = (node_b, "_result")
    root.get_output("result").source = (node_c, "_result")

    # Test actual execution: ft_return3(ft_return2(ft_return1(10))) = ft_return3(ft_return2(11)) = ft_return3(12) = 13
    fused_func = module.create_fused_function(root, "test_chain")
    result = fused_func(10)
    assert result == 10


def test_multiple_inputs_from_same_parent():
    """Test a child node that takes multiple inputs from the parent."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    child = FuseNode.from_function(module.require_function("ft_add"))
    root = FuseNode("root", ["x", "y"], ["out"])

    root.children.append(child)
    child.get_input("a").source = (None, "x")
    child.get_input("b").source = (None, "y")
    root.get_output("out").source = (child, "_result")

    # Test actual execution
    fused_func = module.create_fused_function(root, "test_multiple_inputs")
    result = fused_func(15, 25)
    assert result == 40, f"Expected 40, got {result}"


def test_nested_children():
    """Test code generation with nested children (children that have their own children)."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create leaf nodes (innermost children) using ft_return functions
    leaf1 = FuseNode.from_function(module.require_function("ft_return1"))
    leaf2 = FuseNode.from_function(module.require_function("ft_return2"))

    # Create middle node that has children
    middle = FuseNode("middle", ["x", "y"], ["result"])
    middle.children.append(leaf1)
    middle.children.append(leaf2)
    leaf1.get_input("a").source = (None, "x")
    leaf2.get_input("a").source = (None, "y")
    middle.get_output("result").source = (leaf2, "_result")

    # Create another child at the same level as middle
    sibling = FuseNode.from_function(module.require_function("ft_return3"))

    # Create root that has both middle and sibling as children
    root = FuseNode("root", ["p", "q"], ["final"])
    root.children.append(middle)
    root.children.append(sibling)

    # Connect middle inputs from root
    middle.get_input("x").source = (None, "p")
    middle.get_input("y").source = (None, "q")

    # Connect sibling input from middle output
    sibling.get_input("a").source = (middle, "result")

    # Connect root output from sibling
    root.get_output("final").source = (sibling, "_result")

    # Test actual execution
    fused_func = module.create_fused_function(root, "test_nested")
    result = fused_func(10, 20)
    # ft_return2(20) = 20, ft_return3(20) = 20
    assert result == 20


def test_duplicate_function_reuse():
    """Test that using the same Slang function multiple times doesn't create duplicates."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create two separate add nodes and one multiply node
    add1 = FuseNode.from_function(module.require_function("ft_add"))
    add2 = FuseNode.from_function(module.require_function("ft_add"))
    mul = FuseNode.from_function(module.require_function("ft_mul"))

    # Root takes 4 inputs: a, b, c, d
    root = FuseNode("root", ["a", "b", "c", "d"], ["result"])

    root.children.extend([add1, add2, mul])

    # add1: adds a + b
    add1.get_input("a").source = (None, "a")
    add1.get_input("b").source = (None, "b")

    # add2: adds c + d
    add2.get_input("a").source = (None, "c")
    add2.get_input("b").source = (None, "d")

    # mul: multiplies results of add1 and add2
    mul.get_input("a").source = (add1, "_result")
    mul.get_input("b").source = (add2, "_result")

    # root returns mul result
    root.get_output("result").source = (mul, "_result")

    # Test actual execution
    fused_func = module.create_fused_function(root, "test_duplicate_reuse")
    result = fused_func(5, 10, 2, 3)
    # (5 + 10) * (2 + 3) = 15 * 5 = 75
    assert result == 75, f"Expected 75, got {result}"


def test_subgraph_reuse():
    """Test that subgraphs can be defined once and reused multiple times."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create a subgraph that adds 4 numbers (a + b + c + d)
    # This is the reusable component we want to use twice

    # Build the add4 subgraph internals
    add1 = FuseNode.from_function(module.require_function("ft_add"))
    add2 = FuseNode.from_function(module.require_function("ft_add"))
    add3 = FuseNode.from_function(module.require_function("ft_add"))

    # Create the root of the subgraph
    add4_root = FuseNode("add4", ["a", "b", "c", "d"], ["result"])
    add4_root.children.extend([add1, add2, add3])

    # Wire up: add1 = a + b, add2 = c + d, add3 = add1 + add2
    add1.get_input("a").source = (None, "a")
    add1.get_input("b").source = (None, "b")
    add2.get_input("a").source = (None, "c")
    add2.get_input("b").source = (None, "d")
    add3.get_input("a").source = (add1, "_result")
    add3.get_input("b").source = (add2, "_result")
    add4_root.get_output("result").source = (add3, "_result")

    # Create a FuseSubgraph wrapper
    add4_subgraph = FuseSubgraph("add4", add4_root)

    # Now create two instances of this subgraph
    add4_instance1 = FuseNode.from_subgraph(add4_subgraph)
    add4_instance2 = FuseNode.from_subgraph(add4_subgraph)

    # Create a multiply node
    mul = FuseNode.from_function(module.require_function("ft_mul"))

    # Create root that uses both instances and multiplies results
    root = FuseNode("root", ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8"], ["final"])
    root.children.extend([add4_instance1, add4_instance2, mul])

    # First instance adds n1+n2+n3+n4
    add4_instance1.get_input("a").source = (None, "n1")
    add4_instance1.get_input("b").source = (None, "n2")
    add4_instance1.get_input("c").source = (None, "n3")
    add4_instance1.get_input("d").source = (None, "n4")

    # Second instance adds n5+n6+n7+n8
    add4_instance2.get_input("a").source = (None, "n5")
    add4_instance2.get_input("b").source = (None, "n6")
    add4_instance2.get_input("c").source = (None, "n7")
    add4_instance2.get_input("d").source = (None, "n8")

    # Multiply the two sums
    mul.get_input("a").source = (add4_instance1, "result")
    mul.get_input("b").source = (add4_instance2, "result")

    root.get_output("final").source = (mul, "_result")

    # Test actual execution
    fused_func = module.create_fused_function(root, "test_subgraph_reuse")
    result = fused_func(1, 2, 3, 4, 5, 6, 7, 8)
    # (1+2+3+4) * (5+6+7+8) = 10 * 26 = 260
    assert result == 260, f"Expected 260, got {result}"


def test_nested_subgraphs():
    """Test that subgraphs can contain other subgraphs."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create a subgraph that adds 2 numbers
    add1 = FuseNode.from_function(module.require_function("ft_add"))
    add2_root = FuseNode("add2", ["x", "y"], ["sum"])
    add2_root.children.append(add1)
    add1.get_input("a").source = (None, "x")
    add1.get_input("b").source = (None, "y")
    add2_root.get_output("sum").source = (add1, "_result")
    add2_subgraph = FuseSubgraph("add2", add2_root)

    # Create a subgraph that uses add2 three times to add 4 numbers
    add2_inst1 = FuseNode.from_subgraph(add2_subgraph)
    add2_inst2 = FuseNode.from_subgraph(add2_subgraph)
    add2_inst3 = FuseNode.from_subgraph(add2_subgraph)

    add4_root = FuseNode("add4_nested", ["a", "b", "c", "d"], ["result"])
    add4_root.children.extend([add2_inst1, add2_inst2, add2_inst3])

    # First instance adds a + b
    add2_inst1.get_input("x").source = (None, "a")
    add2_inst1.get_input("y").source = (None, "b")

    # Second instance adds c + d
    add2_inst2.get_input("x").source = (None, "c")
    add2_inst2.get_input("y").source = (None, "d")

    # Third instance adds results of first two
    add2_inst3.get_input("x").source = (add2_inst1, "sum")
    add2_inst3.get_input("y").source = (add2_inst2, "sum")

    add4_root.get_output("result").source = (add2_inst3, "sum")
    add4_subgraph = FuseSubgraph("add4_nested", add4_root)

    # Use the nested subgraph in a root
    add4_inst = FuseNode.from_subgraph(add4_subgraph)
    root = FuseNode("root", ["n1", "n2", "n3", "n4"], ["final"])
    root.children.append(add4_inst)

    add4_inst.get_input("a").source = (None, "n1")
    add4_inst.get_input("b").source = (None, "n2")
    add4_inst.get_input("c").source = (None, "n3")
    add4_inst.get_input("d").source = (None, "n4")

    root.get_output("final").source = (add4_inst, "result")

    # Test actual execution
    fused_func = module.create_fused_function(root, "test_nested_subgraphs")
    result = fused_func(10, 20, 30, 40)
    # (10 + 20) + (30 + 40) = 30 + 70 = 100
    assert result == 100, f"Expected 100, got {result}"


def test_call_simple_fused_function():
    """Test calling a simple fused function through the module API."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create a simple fused function that just wraps ft_add
    node = FuseNode.from_function(module.require_function("ft_add"))
    fused_func = module.create_fused_function(node, "my_add")

    # Try calling it with scalar arguments
    result = fused_func(5, 3)
    assert result == 8, f"Expected 8, got {result}"


def test_call_generic_fused_function():
    """Test calling a simple fused function through the module API."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create a simple fused function that just wraps ft_generic_add
    node = FuseNode.from_function(module.require_function("ft_generic_add"))
    fused_func = module.create_fused_function(node, "generic_add")

    # Try calling it with scalar arguments
    result = fused_func(5, 3)
    assert result == 8, f"Expected 8, got {result}"
