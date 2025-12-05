# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from typing import Optional
import slangpy as spy
import slangpy.testing.helpers as helpers


class FuseSubgraph:
    """
    Represents a reusable subgraph that can be instantiated multiple times.
    This is the template/definition of a subgraph, not an instance.
    """

    def __init__(self, name: str, root_node: "FuseNode"):
        self.name = name
        self.root_node = root_node
        # Input/output names are derived from the root node
        self.input_names = root_node.input_names
        self.output_names = root_node.output_names


class FuseNode:
    def __init__(self, name: str, input_names: list[str], output_names: list[str]):
        super().__init__()
        self.name = name
        self.input_names = input_names
        self.output_names = output_names
        self.function: Optional[spy.Function] = None
        self.subgraph: Optional[FuseSubgraph] = None
        self.children: list["FuseNode"] = []
        # Maps each input name to its source: either (child_node, output_name) or (None, parent_input_name)
        self.input_sources: dict[str, tuple[Optional["FuseNode"], str]] = {}
        # Maps each output name to its source: (child_node, output_name) or None if not connected
        self.output_sources: dict[str, tuple[Optional["FuseNode"], str]] = {}
        # Type information (populated by infer_types pass)
        self.input_types: dict[str, str] = {}  # Maps input name to type string
        self.output_types: dict[str, str] = {}  # Maps output name to type string

    @staticmethod
    def from_function(function: spy.Function) -> "FuseNode":
        assert not function._slang_func.is_overloaded
        res = FuseNode(f"__call_{function.name}", [], [])
        res.function = function
        res.input_names = [arg.name for arg in function._slang_func.parameters]
        res.output_names = ["_result"]
        return res

    @staticmethod
    def from_subgraph(subgraph: FuseSubgraph) -> "FuseNode":
        """Create a node that references a subgraph (for reuse)."""
        res = FuseNode(subgraph.name, subgraph.input_names.copy(), subgraph.output_names.copy())
        res.subgraph = subgraph
        return res


def infer_types(node: FuseNode, inferred_nodes: Optional[set[int]] = None) -> None:
    """
    First pass: Infer types for all nodes in the tree.

    For leaf nodes (with functions), extract types from the Slang function.
    For non-leaf nodes, infer types by tracing through the graph.

    Args:
        node: The node to infer types for
        inferred_nodes: Set of node IDs that have already been processed (to avoid duplicates)
    """
    if inferred_nodes is None:
        inferred_nodes = set()

    # Skip if we've already processed this node
    node_id = id(node)
    if node_id in inferred_nodes:
        return
    inferred_nodes.add(node_id)

    # Process children first (bottom-up inference)
    for child in node.children:
        infer_types(child, inferred_nodes)

    # If this node references a subgraph, infer types for the subgraph root
    if node.subgraph is not None:
        infer_types(node.subgraph.root_node, inferred_nodes)
        # Copy types from subgraph root
        node.input_types = node.subgraph.root_node.input_types.copy()
        node.output_types = node.subgraph.root_node.output_types.copy()
        return

    # For leaf nodes with functions, extract types from the Slang function
    if node.function is not None:
        # Get input types from function parameters
        for param in node.function._slang_func.parameters:
            node.input_types[param.name] = param.type.full_name

        # Get return type
        return_type = node.function._slang_func.return_type.full_name
        for output_name in node.output_names:
            node.output_types[output_name] = return_type
        return

    # For non-leaf nodes, infer types by tracing through the graph
    # Infer input types
    for input_name in node.input_names:
        # Input types are determined by where they're used
        # We'll trace through children to find what type is expected
        type_found = False
        for child in node.children:
            for child_input_name in child.input_names:
                if child_input_name in child.input_sources:
                    source_node, source_output = child.input_sources[child_input_name]
                    if source_node is None and source_output == input_name:
                        # This child uses our input
                        if child_input_name in child.input_types:
                            node.input_types[input_name] = child.input_types[child_input_name]
                            type_found = True
                            break
            if type_found:
                break

    # Infer output types
    for output_name in node.output_names:
        if output_name in node.output_sources:
            source_node, source_output = node.output_sources[output_name]
            if source_node is not None and source_output in source_node.output_types:
                node.output_types[output_name] = source_node.output_types[source_output]


def generate_code(
    node: FuseNode,
    generated_function_names: Optional[set[str]] = None,
    generated_subgraphs: Optional[set[str]] = None,
    infer_types_pass: bool = True,
) -> str:
    """
    Generate code for a node and its children.

    Args:
        node: The node to generate code for
        generated_function_names: Set of function names that have already been generated (to avoid duplicates)
        generated_subgraphs: Set of subgraph names that have already been generated (to avoid duplicates)
        infer_types_pass: Whether to run the type inference pass (should be True for root calls)
    """
    if generated_function_names is None:
        generated_function_names = set()
    if generated_subgraphs is None:
        generated_subgraphs = set()

    # Run type inference pass first (only on the initial call)
    if infer_types_pass:
        infer_types(node)

    code_lines = []

    # Generate child node functions recursively, deduplicating by function name and subgraph name
    for child in node.children:
        # Skip if we've already generated a function with this name
        if child.function is not None and child.name in generated_function_names:
            continue
        # Skip if we've already generated a subgraph with this name
        if child.subgraph is not None and child.name in generated_subgraphs:
            continue

        child_code = generate_code(
            child, generated_function_names, generated_subgraphs, infer_types_pass=False
        )
        code_lines.append(child_code)
        code_lines.append("")  # Add blank line between functions

        # Mark this function name or subgraph name as generated
        if child.function is not None:
            generated_function_names.add(child.name)
        elif child.subgraph is not None:
            generated_subgraphs.add(child.name)

    # If this node references a subgraph, just generate the subgraph's root (no wrapper function)
    if node.subgraph is not None:
        # This is a subgraph reference node - ensure the subgraph's root is generated
        if node.subgraph.name not in generated_subgraphs:
            subgraph_code = generate_code(
                node.subgraph.root_node,
                generated_function_names,
                generated_subgraphs,
                infer_types_pass=False,
            )
            code_lines.append(subgraph_code)
            generated_subgraphs.add(node.subgraph.name)
        # Don't generate a wrapper function - just return what we've collected
        return "\n".join(code_lines)

    # Generate function signature with types
    input_params = ", ".join(
        f"{node.input_types.get(name, 'auto')} {name}" for name in node.input_names
    )
    return_type = (
        node.output_types.get(node.output_names[0], "auto") if node.output_names else "void"
    )
    code_lines.append(f"{return_type} {node.name}({input_params})")
    code_lines.append("{")

    # Generate function body
    if len(node.children) > 0:
        # Topologically sort children based on dependencies
        # Build a dependency graph: which children depend on which other children
        child_dependencies = {child: set() for child in node.children}

        for child in node.children:
            for input_name in child.input_names:
                if input_name in child.input_sources:
                    source_node, _ = child.input_sources[input_name]
                    # If source is from another child, add dependency
                    if source_node is not None and source_node in node.children:
                        child_dependencies[child].add(source_node)

        # Perform topological sort
        sorted_children = []
        remaining = set(node.children)
        # Create a mapping from child to its index in the children list for stable sorting
        child_to_index = {child: i for i, child in enumerate(node.children)}

        while remaining:
            # Find nodes with no dependencies in the remaining set
            ready = [child for child in remaining if not (child_dependencies[child] & remaining)]
            if not ready:
                # Circular dependency or error - just use remaining order
                ready = list(remaining)

            # Sort ready nodes by name (then by original index) to ensure deterministic ordering
            # when multiple nodes are ready at the same time.
            # Use original index as secondary key to handle multiple instances of same subgraph.
            ready.sort(key=lambda x: (x.name, child_to_index[x]))

            for child in ready:
                sorted_children.append(child)
                remaining.remove(child)

        # Track which child outputs we've assigned to temps
        temp_counter = 1
        child_output_to_temp = {}  # Maps (child_node, output_name) to temp variable name

        # Generate calls to child nodes in sorted order
        for child in sorted_children:
            # Build argument list for this child
            child_args = []
            for input_name in child.input_names:
                if input_name in child.input_sources:
                    source_node, source_output_name = child.input_sources[input_name]
                    if source_node is not None:
                        # Input comes from another child's output
                        temp_var = child_output_to_temp.get((source_node, source_output_name))
                        if temp_var:
                            child_args.append(temp_var)
                        else:
                            # Shouldn't happen if graph is valid
                            child_args.append(f"<missing:{source_output_name}>")
                    else:
                        # Input comes from parent node's input
                        child_args.append(source_output_name)
                else:
                    # No source specified - shouldn't happen
                    child_args.append(f"<unconnected:{input_name}>")

            # Generate the call and assign to a temp variable (assuming single output for now)
            if len(child.output_names) > 0:
                temp_var = "temp" if temp_counter == 1 else f"temp{temp_counter}"
                temp_counter += 1
                # Store mapping for all outputs of this child
                for output_name in child.output_names:
                    child_output_to_temp[(child, output_name)] = temp_var
                # Get the type of the child's output
                output_type = child.output_types.get(child.output_names[0], "auto")
                code_lines.append(
                    f"\t{output_type} {temp_var} = {child.name}({', '.join(child_args)})"
                )

        # Generate return statement
        for output_name in node.output_names:
            if output_name in node.output_sources:
                source_node, source_output_name = node.output_sources[output_name]
                if source_node is not None:
                    # Output comes from a child
                    temp_var = child_output_to_temp.get((source_node, source_output_name))
                    if temp_var:
                        code_lines.append(f"\treturn {temp_var}")
                    else:
                        code_lines.append(f"\treturn <missing:{source_output_name}>")
                else:
                    # Output comes directly from an input
                    code_lines.append(f"\treturn {source_output_name}")
    else:
        # Leaf node with no children
        if node.function is not None:
            # Node has an associated function - generate a call to it
            func_args = ", ".join(node.input_names)
            code_lines.append(f"\treturn {node.function.name}({func_args});")
        else:
            code_lines.append("\treturn <not worked out yet>")

    code_lines.append("}")

    return "\n".join(code_lines)


def compare_code(actual: str, expected: str) -> bool:
    """
    Compare two code strings by trimming whitespace and comparing line-by-line.
    Ignores empty lines and is lenient about leading/trailing whitespace on each line.
    Returns True if they match, raises AssertionError with details if not.
    """

    def process_string(s: str) -> list[str]:
        """Trim string, split into lines, filter out empty lines, and trim each line."""
        lines = s.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

    actual_lines = process_string(actual)
    expected_lines = process_string(expected)

    if len(actual_lines) != len(expected_lines):
        raise AssertionError(
            f"Line count mismatch: expected {len(expected_lines)} lines, got {len(actual_lines)} lines\n"
            f"Expected:\n{expected}\n\nActual:\n{actual}"
        )

    for i, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines)):
        if actual_line != expected_line:
            raise AssertionError(
                f"Line {i+1} mismatch:\n"
                f"  Expected: '{expected_line}'\n"
                f"  Actual:   '{actual_line}'\n\n"
                f"Full expected:\n{expected}\n\nFull actual:\n{actual}"
            )

    return True


def test_simple_leaf_node():
    """Test code generation for a node with no children."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")
    node = FuseNode.from_function(module.require_function("ft_add"))

    code = generate_code(node)

    expected = """
    int __call_ft_add(int a, int b)
    {
        return ft_add(a, b);
    }
    """
    assert compare_code(code, expected)


def test_single_child_node():
    """Test code generation for a node with one child."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    child = FuseNode.from_function(module.require_function("ft_add"))
    root = FuseNode("main", ["x", "y"], ["result"])

    root.children.append(child)
    child.input_sources["a"] = (None, "x")
    child.input_sources["b"] = (None, "y")
    root.output_sources["result"] = (child, "_result")

    code = generate_code(root)

    expected = """
    int __call_ft_add(int a, int b)
    {
        return ft_add(a, b);
    }

    int main(int x, int y)
    {
        int temp = __call_ft_add(x, y)
        return temp
    }
    """
    assert compare_code(code, expected)


def test_two_children_with_dependency():
    """Test code generation with two children where one depends on the other."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    mul_node = FuseNode.from_function(module.require_function("ft_mul"))
    add_node = FuseNode.from_function(module.require_function("ft_add"))
    root = FuseNode("compute", ["p", "q", "r"], ["final"])

    root.children.append(mul_node)
    root.children.append(add_node)

    mul_node.input_sources["a"] = (None, "p")
    mul_node.input_sources["b"] = (None, "q")
    add_node.input_sources["a"] = (mul_node, "_result")
    add_node.input_sources["b"] = (None, "r")
    root.output_sources["final"] = (add_node, "_result")

    code = generate_code(root)

    expected = """
    int __call_ft_mul(int a, int b)
    {
        return ft_mul(a, b);
    }

    int __call_ft_add(int a, int b)
    {
        return ft_add(a, b);
    }

    int compute(int p, int q, int r)
    {
        int temp = __call_ft_mul(p, q)
        int temp2 = __call_ft_add(temp, r)
        return temp2
    }
    """
    assert compare_code(code, expected)


def test_topological_sort_reversed_order():
    """Test that topological sort works even when children are added in reverse dependency order."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create nodes using ft_return1 and ft_return2
    first = FuseNode.from_function(module.require_function("ft_return1"))
    second = FuseNode.from_function(module.require_function("ft_return2"))
    root = FuseNode("root", ["a", "b"], ["final"])

    # Add in REVERSE order (second depends on first, but add second first)
    root.children.append(second)
    root.children.append(first)

    # Set up dependencies: second depends on first
    first.input_sources["a"] = (None, "a")
    second.input_sources["a"] = (first, "_result")
    root.output_sources["final"] = (second, "_result")

    code = generate_code(root)

    # Function definitions appear in the order children are added,
    # but the calls inside root() are topologically sorted
    expected = """
    int __call_ft_return2(int a)
    {
        return ft_return2(a);
    }

    int __call_ft_return1(int a)
    {
        return ft_return1(a);
    }

    int root(int a, auto b)
    {
        int temp = __call_ft_return1(a)
        int temp2 = __call_ft_return2(temp)
        return temp2
    }
    """
    assert compare_code(code, expected)


def test_independent_children():
    """Test code generation with two independent children (no dependencies between them)."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    child1 = FuseNode.from_function(module.require_function("ft_return1"))
    child2 = FuseNode.from_function(module.require_function("ft_return2"))
    root = FuseNode("root", ["a", "b"], ["result"])

    root.children.append(child1)
    root.children.append(child2)

    child1.input_sources["a"] = (None, "a")
    child2.input_sources["a"] = (None, "b")
    # Use output from child1 for simplicity
    root.output_sources["result"] = (child1, "_result")

    code = generate_code(root)

    # For independent children with no dependencies, they are sorted alphabetically by name
    # So __call_ft_return1 is executed before __call_ft_return2
    expected = """
    int __call_ft_return1(int a)
    {
        return ft_return1(a);
    }

    int __call_ft_return2(int a)
    {
        return ft_return2(a);
    }

    int root(int a, int b)
    {
        int temp = __call_ft_return1(a)
        int temp2 = __call_ft_return2(b)
        return temp
    }
    """
    assert compare_code(code, expected)


def test_chain_of_three_nodes():
    """Test a chain of three nodes: A -> B -> C."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    node_a = FuseNode.from_function(module.require_function("ft_return1"))
    node_b = FuseNode.from_function(module.require_function("ft_return2"))
    node_c = FuseNode.from_function(module.require_function("ft_return3"))
    root = FuseNode("root", ["x"], ["result"])

    root.children.extend([node_c, node_b, node_a])  # Add in random order

    node_a.input_sources["a"] = (None, "x")
    node_b.input_sources["a"] = (node_a, "_result")
    node_c.input_sources["a"] = (node_b, "_result")
    root.output_sources["result"] = (node_c, "_result")

    code = generate_code(root)

    # Function definitions appear in order added (c, b, a),
    # but calls in root() are topologically sorted (a, b, c)
    expected = """
    int __call_ft_return3(int a)
    {
        return ft_return3(a);
    }

    int __call_ft_return2(int a)
    {
        return ft_return2(a);
    }

    int __call_ft_return1(int a)
    {
        return ft_return1(a);
    }

    int root(int x)
    {
        int temp = __call_ft_return1(x)
        int temp2 = __call_ft_return2(temp)
        int temp3 = __call_ft_return3(temp2)
        return temp3
    }
    """
    assert compare_code(code, expected)


def test_multiple_inputs_from_same_parent():
    """Test a child node that takes multiple inputs from the parent."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    child = FuseNode.from_function(module.require_function("ft_add"))
    root = FuseNode("root", ["x", "y"], ["out"])

    root.children.append(child)
    child.input_sources["a"] = (None, "x")
    child.input_sources["b"] = (None, "y")
    root.output_sources["out"] = (child, "_result")

    code = generate_code(root)

    expected = """
    int __call_ft_add(int a, int b)
    {
        return ft_add(a, b);
    }

    int root(int x, int y)
    {
        int temp = __call_ft_add(x, y)
        return temp
    }
    """
    assert compare_code(code, expected)


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
    leaf1.input_sources["a"] = (None, "x")
    leaf2.input_sources["a"] = (None, "y")
    middle.output_sources["result"] = (leaf2, "_result")

    # Create another child at the same level as middle
    sibling = FuseNode.from_function(module.require_function("ft_return3"))

    # Create root that has both middle and sibling as children
    root = FuseNode("root", ["p", "q", "r"], ["final"])
    root.children.append(middle)
    root.children.append(sibling)

    # Connect middle inputs from root
    middle.input_sources["x"] = (None, "p")
    middle.input_sources["y"] = (None, "q")

    # Connect sibling input from middle output
    sibling.input_sources["a"] = (middle, "result")

    # Connect root output from sibling
    root.output_sources["final"] = (sibling, "_result")

    code = generate_code(root)

    # Expected: nested children are generated recursively
    # leaf1 and leaf2 are defined first (as children of middle)
    # then middle is defined
    # then sibling is defined
    # finally root is defined with calls to middle and sibling in dependency order
    # Note: leaf1 and leaf2 are independent, so alphabetically ft_return1 comes before ft_return2
    expected = """
    int __call_ft_return1(int a)
    {
        return ft_return1(a);
    }

    int __call_ft_return2(int a)
    {
        return ft_return2(a);
    }

    int middle(int x, int y)
    {
        int temp = __call_ft_return1(x)
        int temp2 = __call_ft_return2(y)
        return temp2
    }

    int __call_ft_return3(int a)
    {
        return ft_return3(a);
    }

    int root(int p, int q, auto r)
    {
        int temp = middle(p, q)
        int temp2 = __call_ft_return3(temp)
        return temp2
    }
    """
    assert compare_code(code, expected)


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
    add1.input_sources["a"] = (None, "a")
    add1.input_sources["b"] = (None, "b")

    # add2: adds c + d
    add2.input_sources["a"] = (None, "c")
    add2.input_sources["b"] = (None, "d")

    # mul: multiplies results of add1 and add2
    mul.input_sources["a"] = (add1, "_result")
    mul.input_sources["b"] = (add2, "_result")

    # root returns mul result
    root.output_sources["result"] = (mul, "_result")

    code = generate_code(root)

    # Should only have ONE __call_ft_add definition, not two
    expected = """
    int __call_ft_add(int a, int b)
    {
        return ft_add(a, b);
    }

    int __call_ft_mul(int a, int b)
    {
        return ft_mul(a, b);
    }

    int root(int a, int b, int c, int d)
    {
        int temp = __call_ft_add(a, b)
        int temp2 = __call_ft_add(c, d)
        int temp3 = __call_ft_mul(temp, temp2)
        return temp3
    }
    """
    assert compare_code(code, expected)


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
    add1.input_sources["a"] = (None, "a")
    add1.input_sources["b"] = (None, "b")
    add2.input_sources["a"] = (None, "c")
    add2.input_sources["b"] = (None, "d")
    add3.input_sources["a"] = (add1, "_result")
    add3.input_sources["b"] = (add2, "_result")
    add4_root.output_sources["result"] = (add3, "_result")

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
    add4_instance1.input_sources["a"] = (None, "n1")
    add4_instance1.input_sources["b"] = (None, "n2")
    add4_instance1.input_sources["c"] = (None, "n3")
    add4_instance1.input_sources["d"] = (None, "n4")

    # Second instance adds n5+n6+n7+n8
    add4_instance2.input_sources["a"] = (None, "n5")
    add4_instance2.input_sources["b"] = (None, "n6")
    add4_instance2.input_sources["c"] = (None, "n7")
    add4_instance2.input_sources["d"] = (None, "n8")

    # Multiply the two sums
    mul.input_sources["a"] = (add4_instance1, "result")
    mul.input_sources["b"] = (add4_instance2, "result")

    root.output_sources["final"] = (mul, "_result")

    code = generate_code(root)

    # The add4 subgraph should only be defined ONCE, even though used twice
    # The __call_ft_add should also only be defined once
    expected = """
    int __call_ft_add(int a, int b)
    {
        return ft_add(a, b);
    }

    int add4(int a, int b, int c, int d)
    {
        int temp = __call_ft_add(a, b)
        int temp2 = __call_ft_add(c, d)
        int temp3 = __call_ft_add(temp, temp2)
        return temp3
    }

    int __call_ft_mul(int a, int b)
    {
        return ft_mul(a, b);
    }

    int root(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8)
    {
        int temp = add4(n1, n2, n3, n4)
        int temp2 = add4(n5, n6, n7, n8)
        int temp3 = __call_ft_mul(temp, temp2)
        return temp3
    }
    """
    assert compare_code(code, expected)


def test_nested_subgraphs():
    """Test that subgraphs can contain other subgraphs."""
    device = helpers.get_device(spy.DeviceType.d3d12)
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create a subgraph that adds 2 numbers
    add1 = FuseNode.from_function(module.require_function("ft_add"))
    add2_root = FuseNode("add2", ["x", "y"], ["sum"])
    add2_root.children.append(add1)
    add1.input_sources["a"] = (None, "x")
    add1.input_sources["b"] = (None, "y")
    add2_root.output_sources["sum"] = (add1, "_result")
    add2_subgraph = FuseSubgraph("add2", add2_root)

    # Create a subgraph that uses add2 three times to add 4 numbers
    add2_inst1 = FuseNode.from_subgraph(add2_subgraph)
    add2_inst2 = FuseNode.from_subgraph(add2_subgraph)
    add2_inst3 = FuseNode.from_subgraph(add2_subgraph)

    add4_root = FuseNode("add4_nested", ["a", "b", "c", "d"], ["result"])
    add4_root.children.extend([add2_inst1, add2_inst2, add2_inst3])

    # First instance adds a + b
    add2_inst1.input_sources["x"] = (None, "a")
    add2_inst1.input_sources["y"] = (None, "b")

    # Second instance adds c + d
    add2_inst2.input_sources["x"] = (None, "c")
    add2_inst2.input_sources["y"] = (None, "d")

    # Third instance adds results of first two
    add2_inst3.input_sources["x"] = (add2_inst1, "sum")
    add2_inst3.input_sources["y"] = (add2_inst2, "sum")

    add4_root.output_sources["result"] = (add2_inst3, "sum")
    add4_subgraph = FuseSubgraph("add4_nested", add4_root)

    # Use the nested subgraph in a root
    add4_inst = FuseNode.from_subgraph(add4_subgraph)
    root = FuseNode("root", ["n1", "n2", "n3", "n4"], ["final"])
    root.children.append(add4_inst)

    add4_inst.input_sources["a"] = (None, "n1")
    add4_inst.input_sources["b"] = (None, "n2")
    add4_inst.input_sources["c"] = (None, "n3")
    add4_inst.input_sources["d"] = (None, "n4")

    root.output_sources["final"] = (add4_inst, "result")

    code = generate_code(root)

    # Should generate: __call_ft_add, add2, add4_nested, root
    # Note that add2 should only be defined once even though used 3 times
    expected = """
    int __call_ft_add(int a, int b)
    {
        return ft_add(a, b);
    }

    int add2(int x, int y)
    {
        int temp = __call_ft_add(x, y)
        return temp
    }

    int add4_nested(int a, int b, int c, int d)
    {
        int temp = add2(a, b)
        int temp2 = add2(c, d)
        int temp3 = add2(temp, temp2)
        return temp3
    }

    int root(int n1, int n2, int n3, int n4)
    {
        int temp = add4_nested(n1, n2, n3, n4)
        return temp
    }
    """
    assert compare_code(code, expected)
