# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from typing import Optional
import slangpy as spy


class FuseNode:
    def __init__(self, name: str, input_names: list[str], output_names: list[str]):
        super().__init__()
        self.name = name
        self.input_names = input_names
        self.output_names = output_names
        self.function: Optional[spy.Function] = None
        self.children: list["FuseNode"] = []
        # Maps each input name to its source: either (child_node, output_name) or (None, parent_input_name)
        self.input_sources: dict[str, tuple[Optional["FuseNode"], str]] = {}
        # Maps each output name to its source: (child_node, output_name) or None if not connected
        self.output_sources: dict[str, tuple[Optional["FuseNode"], str]] = {}


def generate_code(node: FuseNode) -> str:
    code_lines = []

    # Generate child node functions recursively
    for child in node.children:
        child_code = generate_code(child)
        code_lines.append(child_code)
        code_lines.append("")  # Add blank line between functions

    # Generate function signature
    input_params = ", ".join(node.input_names)
    code_lines.append(f"{node.name}({input_params})")
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

        while remaining:
            # Find nodes with no dependencies in the remaining set
            ready = [child for child in remaining if not (child_dependencies[child] & remaining)]
            if not ready:
                # Circular dependency or error - just use remaining order
                ready = list(remaining)

            # Sort ready nodes by name to ensure deterministic ordering
            # when multiple nodes are ready at the same time
            ready.sort(key=lambda x: x.name)

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
                code_lines.append(f"\t{temp_var} = {child.name}({', '.join(child_args)})")

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
        code_lines.append("\treturn <not worked out yet>")

    code_lines.append("}")

    return "\n".join(code_lines)


def main():
    device = spy.create_device(include_paths=[Path(__file__).parent])
    module = spy.Module.load_from_file(device, "fusetest.slang")

    # Create child nodes
    mul_node = FuseNode("mul_node", ["mul_in_a", "mul_in_b"], ["mul_out"])
    add_node = FuseNode("add_node", ["add_in_a", "add_in_b"], ["add_out"])

    # Create root node
    root = FuseNode("root", ["a", "b", "c"], ["out"])

    # Add children (order doesn't matter due to topological sort)
    root.children.append(add_node)
    root.children.append(mul_node)

    # Connect mul_node inputs to root inputs
    mul_node.input_sources["mul_in_a"] = (None, "a")  # From root input "a"
    mul_node.input_sources["mul_in_b"] = (None, "b")  # From root input "b"

    # Connect add_node inputs: one from mul_node output, one from root input
    add_node.input_sources["add_in_a"] = (mul_node, "mul_out")  # From mul_node output
    add_node.input_sources["add_in_b"] = (None, "c")  # From root input "c"

    # Connect root output to add_node output
    root.output_sources["out"] = (add_node, "add_out")

    print(generate_code(root))


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
    node = FuseNode("simple_func", ["x", "y"], ["result"])

    code = generate_code(node)

    expected = """
    simple_func(x, y)
    {
        return <not worked out yet>
    }
    """
    assert compare_code(code, expected)


def test_single_child_node():
    """Test code generation for a node with one child."""
    child = FuseNode("helper", ["a", "b"], ["out"])
    root = FuseNode("main", ["x", "y"], ["result"])

    root.children.append(child)
    child.input_sources["a"] = (None, "x")
    child.input_sources["b"] = (None, "y")
    root.output_sources["result"] = (child, "out")

    code = generate_code(root)

    expected = """
    helper(a, b)
    {
        return <not worked out yet>
    }

    main(x, y)
    {
        temp = helper(x, y)
        return temp
    }
    """
    assert compare_code(code, expected)


def test_two_children_with_dependency():
    """Test code generation with two children where one depends on the other."""
    mul_node = FuseNode("mul", ["a", "b"], ["result"])
    add_node = FuseNode("add", ["x", "y"], ["result"])
    root = FuseNode("compute", ["p", "q", "r"], ["final"])

    root.children.append(mul_node)
    root.children.append(add_node)

    mul_node.input_sources["a"] = (None, "p")
    mul_node.input_sources["b"] = (None, "q")
    add_node.input_sources["x"] = (mul_node, "result")
    add_node.input_sources["y"] = (None, "r")
    root.output_sources["final"] = (add_node, "result")

    code = generate_code(root)

    expected = """
    mul(a, b)
    {
        return <not worked out yet>
    }

    add(x, y)
    {
        return <not worked out yet>
    }

    compute(p, q, r)
    {
        temp = mul(p, q)
        temp2 = add(temp, r)
        return temp2
    }
    """
    assert compare_code(code, expected)


def test_topological_sort_reversed_order():
    """Test that topological sort works even when children are added in reverse dependency order."""
    # Create nodes
    first = FuseNode("first", ["x"], ["out1"])
    second = FuseNode("second", ["y"], ["out2"])
    root = FuseNode("root", ["a", "b"], ["final"])

    # Add in REVERSE order (second depends on first, but add second first)
    root.children.append(second)
    root.children.append(first)

    # Set up dependencies: second depends on first
    first.input_sources["x"] = (None, "a")
    second.input_sources["y"] = (first, "out1")
    root.output_sources["final"] = (second, "out2")

    code = generate_code(root)

    # Function definitions appear in the order children are added,
    # but the calls inside root() are topologically sorted
    expected = """
    second(y)
    {
        return <not worked out yet>
    }

    first(x)
    {
        return <not worked out yet>
    }

    root(a, b)
    {
        temp = first(a)
        temp2 = second(temp)
        return temp2
    }
    """
    assert compare_code(code, expected)


def test_independent_children():
    """Test code generation with two independent children (no dependencies between them)."""
    child1 = FuseNode("func1", ["x"], ["out1"])
    child2 = FuseNode("func2", ["y"], ["out2"])
    root = FuseNode("root", ["a", "b"], ["result"])

    root.children.append(child1)
    root.children.append(child2)

    child1.input_sources["x"] = (None, "a")
    child2.input_sources["y"] = (None, "b")
    # Use output from child1 for simplicity
    root.output_sources["result"] = (child1, "out1")

    code = generate_code(root)

    # For independent children with no dependencies, they are sorted alphabetically by name
    # So func1 is executed before func2
    expected = """
    func1(x)
    {
        return <not worked out yet>
    }

    func2(y)
    {
        return <not worked out yet>
    }

    root(a, b)
    {
        temp = func1(a)
        temp2 = func2(b)
        return temp
    }
    """
    assert compare_code(code, expected)


def test_chain_of_three_nodes():
    """Test a chain of three nodes: A -> B -> C."""
    node_a = FuseNode("node_a", ["in"], ["out"])
    node_b = FuseNode("node_b", ["in"], ["out"])
    node_c = FuseNode("node_c", ["in"], ["out"])
    root = FuseNode("root", ["x"], ["result"])

    root.children.extend([node_c, node_b, node_a])  # Add in random order

    node_a.input_sources["in"] = (None, "x")
    node_b.input_sources["in"] = (node_a, "out")
    node_c.input_sources["in"] = (node_b, "out")
    root.output_sources["result"] = (node_c, "out")

    code = generate_code(root)

    # Function definitions appear in order added (c, b, a),
    # but calls in root() are topologically sorted (a, b, c)
    expected = """
    node_c(in)
    {
        return <not worked out yet>
    }

    node_b(in)
    {
        return <not worked out yet>
    }

    node_a(in)
    {
        return <not worked out yet>
    }

    root(x)
    {
        temp = node_a(x)
        temp2 = node_b(temp)
        temp3 = node_c(temp2)
        return temp3
    }
    """
    assert compare_code(code, expected)


def test_multiple_inputs_from_same_parent():
    """Test a child node that takes multiple inputs from the parent."""
    child = FuseNode("combine", ["x", "y", "z"], ["result"])
    root = FuseNode("root", ["a", "b", "c"], ["out"])

    root.children.append(child)
    child.input_sources["x"] = (None, "a")
    child.input_sources["y"] = (None, "b")
    child.input_sources["z"] = (None, "c")
    root.output_sources["out"] = (child, "result")

    code = generate_code(root)

    expected = """
    combine(x, y, z)
    {
        return <not worked out yet>
    }

    root(a, b, c)
    {
        temp = combine(a, b, c)
        return temp
    }
    """
    assert compare_code(code, expected)


def test_nested_children():
    """Test code generation with nested children (children that have their own children)."""
    # Create leaf nodes (innermost children)
    leaf1 = FuseNode("leaf1", ["x"], ["out"])
    leaf2 = FuseNode("leaf2", ["y"], ["out"])

    # Create middle node that has children
    middle = FuseNode("middle", ["a", "b"], ["result"])
    middle.children.append(leaf1)
    middle.children.append(leaf2)
    leaf1.input_sources["x"] = (None, "a")
    leaf2.input_sources["y"] = (None, "b")
    middle.output_sources["result"] = (leaf2, "out")

    # Create another child at the same level as middle
    sibling = FuseNode("sibling", ["z"], ["out"])

    # Create root that has both middle and sibling as children
    root = FuseNode("root", ["p", "q", "r"], ["final"])
    root.children.append(middle)
    root.children.append(sibling)

    # Connect middle inputs from root
    middle.input_sources["a"] = (None, "p")
    middle.input_sources["b"] = (None, "q")

    # Connect sibling input from middle output
    sibling.input_sources["z"] = (middle, "result")

    # Connect root output from sibling
    root.output_sources["final"] = (sibling, "out")

    code = generate_code(root)

    # Expected: nested children are generated recursively
    # leaf1 and leaf2 are defined first (as children of middle)
    # then middle is defined
    # then sibling is defined
    # finally root is defined with calls to middle and sibling in dependency order
    # Note: leaf1 and leaf2 are independent, so their execution order can vary
    expected = """
    leaf1(x)
    {
        return <not worked out yet>
    }

    leaf2(y)
    {
        return <not worked out yet>
    }

    middle(a, b)
    {
        temp = leaf1(a)
        temp2 = leaf2(b)
        return temp2
    }

    sibling(z)
    {
        return <not worked out yet>
    }

    root(p, q, r)
    {
        temp = middle(p, q)
        temp2 = sibling(temp)
        return temp2
    }
    """
    assert compare_code(code, expected)


if __name__ == "__main__":
    main()
