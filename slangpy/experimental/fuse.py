# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import slangpy as spy


class FusePort:
    """
    Represents information about an input or output port.
    Contains name, type, and source information.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.type: str = "auto"  # Type string, defaults to "auto"
        # Source: (source_node, source_port_name) or (None, parent_port_name)
        self.source: Optional[tuple[Optional["FuseNode"], str]] = None


class FuseNode:
    """Represents a node in the fusion graph."""

    def __init__(self, name: str, input_names: list[str], output_names: list[str]):
        super().__init__()
        self.name = name
        self.inputs = [FusePort(name) for name in input_names]
        self.outputs = [FusePort(name) for name in output_names]
        self.function: Optional[spy.Function] = None
        self.subgraph: Optional["FuseSubgraph"] = None
        self.children: list["FuseNode"] = []

    def get_input(self, name: str) -> FusePort:
        """Get input port by name."""
        for port in self.inputs:
            if port.name == name:
                return port
        raise ValueError(f"Input port '{name}' not found in node '{self.name}'")

    def get_output(self, name: str) -> FusePort:
        """Get output port by name."""
        for port in self.outputs:
            if port.name == name:
                return port
        raise ValueError(f"Output port '{name}' not found in node '{self.name}'")

    @staticmethod
    def from_function(function: spy.Function) -> "FuseNode":
        """Create a node from a Slang function."""
        assert not function._slang_func.is_overloaded
        input_names = [arg.name for arg in function._slang_func.parameters]
        res = FuseNode(f"__call_{function.name}", input_names, ["_result"])
        res.function = function
        return res

    @staticmethod
    def from_subgraph(subgraph: "FuseSubgraph") -> "FuseNode":
        """Create a node that references a subgraph (for reuse)."""
        input_names = [port.name for port in subgraph.root_node.inputs]
        output_names = [port.name for port in subgraph.root_node.outputs]
        res = FuseNode(subgraph.name, input_names, output_names)
        res.subgraph = subgraph
        return res


class FuseSubgraph:
    """
    Represents a reusable subgraph that can be instantiated multiple times.
    This is the template/definition of a subgraph, not an instance.
    """

    def __init__(self, name: str, root_node: FuseNode):
        super().__init__()
        self.name = name
        self.root_node = root_node


class Fuser:
    """Main class for managing and generating code from a fusion graph."""

    def __init__(self, root_node: FuseNode):
        super().__init__()
        self.root_node = root_node

    def generate_code(self) -> str:
        """
        Public API: Generate code for the fusion graph.

        Returns:
            Generated Slang code as a string.
        """
        # Run type inference first
        self._infer_types(self.root_node)

        # Generate code
        generated_function_names: set[str] = set()
        generated_subgraphs: set[str] = set()
        return self._generate_code(self.root_node, generated_function_names, generated_subgraphs)

    def _infer_types(self, node: FuseNode, inferred_nodes: Optional[set[int]] = None) -> None:
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
            self._infer_types(child, inferred_nodes)

        # If this node references a subgraph, infer types for the subgraph root
        if node.subgraph is not None:
            self._infer_types(node.subgraph.root_node, inferred_nodes)
            # Copy types from subgraph root
            for i, port in enumerate(node.inputs):
                port.type = node.subgraph.root_node.inputs[i].type
            for i, port in enumerate(node.outputs):
                port.type = node.subgraph.root_node.outputs[i].type
            return

        # For leaf nodes with functions, extract types from the Slang function
        if node.function is not None:
            # Get input types from function parameters
            for i, param in enumerate(node.function._slang_func.parameters):
                node.inputs[i].type = param.type.full_name

            # Get return type
            return_type = node.function._slang_func.return_type.full_name
            for port in node.outputs:
                port.type = return_type
            return

        # For non-leaf nodes, infer types by tracing through the graph
        # Infer input types
        for input_port in node.inputs:
            # Input types are determined by where they're used
            # We'll trace through children to find what type is expected
            type_found = False
            for child in node.children:
                for child_input in child.inputs:
                    if child_input.source is not None:
                        source_node, source_port_name = child_input.source
                        if source_node is None and source_port_name == input_port.name:
                            # This child uses our input
                            input_port.type = child_input.type
                            type_found = True
                            break
                if type_found:
                    break

        # Infer output types
        for output_port in node.outputs:
            if output_port.source is not None:
                source_node, source_port_name = output_port.source
                if source_node is not None:
                    source_output = source_node.get_output(source_port_name)
                    if source_output is not None:
                        output_port.type = source_output.type

    def _sort_graph(self, node: FuseNode) -> list[FuseNode]:
        """
        Topologically sort children based on dependencies.

        Args:
            node: The parent node whose children should be sorted

        Returns:
            List of children in topologically sorted order
        """
        # Build a dependency graph: which children depend on which other children
        child_dependencies = {child: set() for child in node.children}

        for child in node.children:
            for input_port in child.inputs:
                if input_port.source is not None:
                    source_node, _ = input_port.source
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

        return sorted_children

    def _generate_code(
        self,
        node: FuseNode,
        generated_function_names: set[str],
        generated_subgraphs: set[str],
    ) -> str:
        """
        Generate code for a node and its children.

        Args:
            node: The node to generate code for
            generated_function_names: Set of function names that have already been generated
            generated_subgraphs: Set of subgraph names that have already been generated

        Returns:
            Generated code as a string
        """
        code_lines = []

        # Generate child node functions recursively, deduplicating by function name and subgraph name
        for child in node.children:
            # Skip if we've already generated a function with this name
            if child.function is not None and child.name in generated_function_names:
                continue
            # Skip if we've already generated a subgraph with this name
            if child.subgraph is not None and child.name in generated_subgraphs:
                continue

            child_code = self._generate_code(child, generated_function_names, generated_subgraphs)
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
                subgraph_code = self._generate_code(
                    node.subgraph.root_node, generated_function_names, generated_subgraphs
                )
                code_lines.append(subgraph_code)
                generated_subgraphs.add(node.subgraph.name)
            # Don't generate a wrapper function - just return what we've collected
            return "\n".join(code_lines)

        # Generate the function signature and body
        signature = self._generate_function_signature(node)
        body = self._generate_function_body(node)

        code_lines.append(signature)
        code_lines.append("{")
        code_lines.append(body)
        code_lines.append("}")

        return "\n".join(code_lines)

    def _generate_function_signature(self, node: FuseNode) -> str:
        """
        Generate the function signature for a node.

        Args:
            node: The node to generate signature for

        Returns:
            Function signature string
        """
        input_params = ", ".join(f"{port.type} {port.name}" for port in node.inputs)
        return_type = node.outputs[0].type if node.outputs else "void"
        return f"{return_type} {node.name}({input_params})"

    def _generate_function_body(self, node: FuseNode) -> str:
        """
        Generate the function body for a node.

        Args:
            node: The node to generate body for

        Returns:
            Function body string (without braces)
        """
        body_lines = []

        if len(node.children) > 0:
            # Sort children topologically
            sorted_children = self._sort_graph(node)

            # Track which child outputs we've assigned to temps
            temp_counter = 1
            child_output_to_temp = {}  # Maps (child_node, output_name) to temp variable name

            # Generate calls to child nodes in sorted order
            for child in sorted_children:
                # Build argument list for this child
                child_args = []
                for input_port in child.inputs:
                    if input_port.source is not None:
                        source_node, source_output_name = input_port.source
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
                        child_args.append(f"<unconnected:{input_port.name}>")

                # Generate the call and assign to a temp variable (assuming single output for now)
                if len(child.outputs) > 0:
                    temp_var = "temp" if temp_counter == 1 else f"temp{temp_counter}"
                    temp_counter += 1
                    # Store mapping for all outputs of this child
                    for output_port in child.outputs:
                        child_output_to_temp[(child, output_port.name)] = temp_var
                    # Get the type of the child's output
                    output_type = child.outputs[0].type
                    body_lines.append(
                        f"\t{output_type} {temp_var} = {child.name}({', '.join(child_args)})"
                    )

            # Generate return statement
            for output_port in node.outputs:
                if output_port.source is not None:
                    source_node, source_output_name = output_port.source
                    if source_node is not None:
                        # Output comes from a child
                        temp_var = child_output_to_temp.get((source_node, source_output_name))
                        if temp_var:
                            body_lines.append(f"\treturn {temp_var}")
                        else:
                            body_lines.append(f"\treturn <missing:{source_output_name}>")
                    else:
                        # Output comes directly from an input
                        body_lines.append(f"\treturn {source_output_name}")
        else:
            # Leaf node with no children
            if node.function is not None:
                # Node has an associated function - generate a call to it
                func_args = ", ".join(port.name for port in node.inputs)
                body_lines.append(f"\treturn {node.function.name}({func_args});")
            else:
                body_lines.append("\treturn <not worked out yet>")

        return "\n".join(body_lines)
