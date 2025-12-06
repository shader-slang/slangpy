# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
from typing import TYPE_CHECKING, Optional, Any
from slangpy.reflection.reflectiontypes import SlangType
from slangpy import ModifierID

if TYPE_CHECKING:
    from slangpy.core.module import Module
    from slangpy.bindings.codegen import CodeGen
    from slangpy.core.function import Function
    from slangpy.bindings.boundvariable import BoundVariable


def _sort_graph(node: "FuseNode") -> list["FuseNode"]:
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


class FusePort:
    """
    Represents information about an input or output port.
    Contains name, type, and source information.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.type: Optional[SlangType] = None  # SlangType object, None means not yet inferred
        # Source: (source_node, source_port_name) or (None, parent_port_name)
        self.source: Optional[tuple[Optional["FuseNode"], str]] = None
        self.binding: Optional["BoundVariable"] = None  # For future use with calldata.py

    def __deepcopy__(self, memo: Any) -> "FusePort":
        new_port = FusePort(self.name)
        memo[id(self)] = new_port
        new_port.type = self.type
        new_port.source = copy.deepcopy(self.source, memo)
        return new_port


class FuseNode:
    """Represents a node in the fusion graph."""

    def __init__(self, name: str, input_names: list[str], output_names: list[str]):
        super().__init__()
        self.name = name
        self.inputs = [FusePort(name) for name in input_names]
        self.outputs = [FusePort(name) for name in output_names]
        self.function: Optional["Function"] = None
        self.subgraph: Optional["FuseSubgraph"] = None
        self.children: list["FuseNode"] = []

    def __deepcopy__(self, memo: Any) -> "FuseNode":
        new_node = FuseNode(self.name, [], [])
        memo[id(self)] = new_node

        new_node.inputs = [copy.deepcopy(port, memo) for port in self.inputs]
        new_node.outputs = [copy.deepcopy(port, memo) for port in self.outputs]
        new_node.function = self.function  # Functions are immutable, safe to share
        new_node.subgraph = (
            copy.deepcopy(self.subgraph, memo) if self.subgraph is not None else None
        )
        new_node.children = [copy.deepcopy(child, memo) for child in self.children]

        return new_node

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
    def from_function(function) -> "FuseNode":
        """Create a node from a Slang function."""
        # Import at runtime to avoid circular import
        import slangpy

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

    def validate_types(self) -> None:
        """
        Validate that all ports have types assigned.
        Raises ValueError if any port is missing a type.
        """
        for port in self.inputs:
            if port.type is None:
                raise ValueError(
                    f"Input port '{port.name}' in node '{self.name}' has no type. "
                    "Run type inference first."
                )
        for port in self.outputs:
            if port.type is None:
                raise ValueError(
                    f"Output port '{port.name}' in node '{self.name}' has no type. "
                    "Run type inference first."
                )

    def to_function(self, module: "Module", name: Optional[str] = None) -> "FusedFunction":
        """
        Convert this FuseNode into a callable FusedFunction.

        Args:
            module: The module to associate with the fused function
            name: Optional name for the function (defaults to node name)

        Returns:
            A FusedFunction that can be used with calldata.py

        Raises:
            ValueError: If type inference is incomplete
        """
        if name is None:
            name = self.name
        fuser = Fuser(self)
        # Run type inference
        fuser._infer_types(self)
        # Now validate
        self.validate_types()
        return FusedFunction(fuser, module, name)


class FuseSubgraph:
    """
    Represents a reusable subgraph that can be instantiated multiple times.
    This is the template/definition of a subgraph, not an instance.
    """

    def __init__(self, name: str, root_node: FuseNode):
        super().__init__()
        self.name = name
        self.root_node = root_node


class FusedParameter:
    """
    Adapter that makes a FusePort compatible with SlangParameter interface.
    This allows fused function parameters to work with calldata.py's binding system.
    """

    def __init__(self, port: FusePort, index: int):
        super().__init__()
        self._port = port
        self._index = index

    @property
    def name(self) -> str:
        """Name of the parameter."""
        return self._port.name

    @property
    def type(self) -> Optional[SlangType]:
        """Type of the parameter."""
        return self._port.type

    @property
    def modifiers(self) -> set[ModifierID]:
        """Modifiers for the parameter (empty for fused functions)."""
        return set()

    @property
    def index(self) -> int:
        """Index of the parameter in the function signature."""
        return self._index


class FusedFunction:
    """
    Adapter that makes a FuseNode compatible with SlangFunction interface.
    This allows fused functions to work with calldata.py's kernel generation system.
    """

    def __init__(self, fuser: "Fuser", module: "Module", name: str):
        super().__init__()
        self._fuser = fuser
        self._module = module
        self._name = name
        self._cached_parameters: Optional[tuple[FusedParameter, ...]] = None
        self._cached_return_type: Optional[SlangType] = None

    def __deepcopy__(self, memo: Any) -> "FusedFunction":
        new_func = FusedFunction(self._fuser, self._module, self._name)
        memo[id(self)] = new_func
        new_func._fuser = copy.deepcopy(self._fuser, memo)
        return new_func

    @property
    def name(self) -> str:
        """Name of the fused function."""
        return self._name

    @property
    def full_name(self) -> str:
        """Fully qualified name (same as name for fused functions)."""
        return self._name

    @property
    def parameters(self) -> tuple[FusedParameter, ...]:
        """Parameters of the fused function."""
        if self._cached_parameters is None:
            params = []
            for i, port in enumerate(self._fuser.root_node.inputs):
                params.append(FusedParameter(port, i))
            self._cached_parameters = tuple(params)
        return self._cached_parameters

    @property
    def return_type(self) -> Optional[SlangType]:
        """Return type of the fused function."""
        if self._cached_return_type is None:
            # Get the first output port's type
            if len(self._fuser.root_node.outputs) > 0:
                self._cached_return_type = self._fuser.root_node.outputs[0].type
        return self._cached_return_type

    @property
    def have_return_value(self) -> bool:
        """Return true if this function doesn't return void."""
        ret_type = self.return_type
        if ret_type is None:
            return False
        return ret_type.name != "void"

    @property
    def differentiable(self) -> bool:
        """Whether this function is differentiable (not supported yet for fused functions)."""
        return False

    @property
    def mutating(self) -> bool:
        """Whether this function is mutating (always False for fused functions)."""
        return False

    @property
    def static(self) -> bool:
        """Whether this function is static (always True for fused functions)."""
        return True

    @property
    def is_overloaded(self) -> bool:
        """Whether this function is overloaded (always False for fused functions)."""
        return False

    @property
    def this(self) -> Optional[SlangType]:
        """Type that this function is a method of (always None for fused functions)."""
        return None

    @property
    def overloads(self) -> tuple["FusedFunction", ...]:
        """Overloads of this function (always empty for fused functions)."""
        return ()

    @property
    def reflection(self):
        """
        Mock reflection object. Note: This is a minimal implementation and may not
        support all operations that a real FunctionReflection would.
        """
        # Return None for now - we'll handle this specially in calldata.py if needed
        return None

    @property
    def is_constructor(self) -> bool:
        """Whether this function is a constructor (always False for fused functions)."""
        return False


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

    def sort_graph(self):
        """
        Public API: Sort the fusion graph topologically.

        This modifies the root node's children in place.
        """

        def _sort_node_rec(node: FuseNode):
            node.children = _sort_graph(node)
            for child in node.children:
                if child.subgraph is not None:
                    _sort_node_rec(child.subgraph.root_node)

        _sort_node_rec(self.root_node)

    def clear_type_info(self) -> None:
        """
        Public API: Recursively clear all type information from the fusion graph.

        This removes all inferred types from all ports in the graph and subgraphs.
        Useful for resetting the graph state or debugging type inference.
        """
        self._clear_type_info_recursive(self.root_node, set())

    def dump_graph(self) -> str:
        """
        Public API: Generate a human-readable string representation of the fusion graph.

        Returns:
            A multi-line string describing the graph structure, nodes, connections,
            and type information (if available).
        """
        lines = []
        lines.append("Fusion Graph:")
        lines.append("=" * 80)
        self._dump_node_recursive(self.root_node, lines, indent=0, visited=set())
        return "\n".join(lines)

    def inject_code_into_codegen(self, cg: "CodeGen", function_name: Optional[str] = None) -> None:
        """
        Inject the fused function code into a CodeGen object.
        This is called by callsignature.py during kernel generation.

        Args:
            cg: CodeGen object to inject code into
            function_name: Optional name to use for the root function (overrides node name)
        """
        # Run type inference if not already done
        self._infer_types(self.root_node)

        # Temporarily override the root node's name if a function name is provided
        original_name = self.root_node.name
        if function_name is not None:
            self.root_node.name = function_name

        try:
            # Generate the code
            generated_function_names: set[str] = set()
            generated_subgraphs: set[str] = set()
            code = self._generate_code(
                self.root_node, generated_function_names, generated_subgraphs
            )

            # Add the generated code as a snippet
            cg.add_snippet(f"fused_function_{self.root_node.name}", code)

            # Track any imports needed for child functions
            self._collect_imports(self.root_node, cg, set())
        finally:
            # Restore the original name
            self.root_node.name = original_name

    def _collect_imports(self, node: FuseNode, cg: "CodeGen", visited: set[int]) -> None:
        """
        Recursively collect all imports needed for child functions.

        Args:
            node: Node to collect imports from
            cg: CodeGen object to add imports to
            visited: Set of already visited node IDs
        """
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        # If this node has a function, add its module as an import
        if node.function is not None:
            module_name = node.function.module.name
            cg.add_import(module_name)

        # If this node has a subgraph, collect its imports
        if node.subgraph is not None:
            self._collect_imports(node.subgraph.root_node, cg, visited)

        # Recursively collect imports from children
        for child in node.children:
            self._collect_imports(child, cg, visited)

    def _clear_type_info_recursive(self, node: FuseNode, visited: set[int]) -> None:
        """
        Recursively clear all type information from a node and its children.

        Args:
            node: Node to clear type info from
            visited: Set of already visited node IDs to avoid infinite recursion
        """
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        # Clear type info from all input ports
        for port in node.inputs:
            port.type = None

        # Clear type info from all output ports
        for port in node.outputs:
            port.type = None

        # If this node has a subgraph, clear its types too
        if node.subgraph is not None:
            self._clear_type_info_recursive(node.subgraph.root_node, visited)

        # Recursively clear types from children
        for child in node.children:
            self._clear_type_info_recursive(child, visited)

    def _dump_node_recursive(
        self, node: FuseNode, lines: list[str], indent: int, visited: set[int]
    ) -> None:
        """
        Recursively generate a human-readable dump of a node and its children.

        Args:
            node: Node to dump
            lines: List of strings to append output to
            indent: Current indentation level
            visited: Set of already visited node IDs to avoid infinite recursion
        """
        node_id = id(node)
        indent_str = "  " * indent

        # Check if we've already visited this node
        if node_id in visited:
            lines.append(f"{indent_str}[Node '{node.name}' - ALREADY VISITED]")
            return
        visited.add(node_id)

        # Node header
        node_type = (
            "SubgraphRef"
            if node.subgraph is not None
            else "Function" if node.function is not None else "Composite"
        )
        lines.append(f"{indent_str}Node: '{node.name}' (type={node_type})")

        # Input ports
        if node.inputs:
            lines.append(f"{indent_str}  Inputs:")
            for port in node.inputs:
                type_str = port.type.full_name if port.type else "<?>"
                source_str = ""
                if port.source is not None:
                    source_node, source_port = port.source
                    if source_node is not None:
                        source_str = f" <- {source_node.name}.{source_port}"
                    else:
                        source_str = f" <- parent.{source_port}"
                lines.append(f"{indent_str}    - {port.name}: {type_str}{source_str}")

        # Output ports
        if node.outputs:
            lines.append(f"{indent_str}  Outputs:")
            for port in node.outputs:
                type_str = port.type.full_name if port.type else "<?>"
                source_str = ""
                if port.source is not None:
                    source_node, source_port = port.source
                    if source_node is not None:
                        source_str = f" <- {source_node.name}.{source_port}"
                    else:
                        source_str = f" <- parent.{source_port}"
                lines.append(f"{indent_str}    - {port.name}: {type_str}{source_str}")

        # Function info
        if node.function is not None:
            lines.append(f"{indent_str}  Function: {node.function.name}")

        # Subgraph reference
        if node.subgraph is not None:
            lines.append(f"{indent_str}  Subgraph: '{node.subgraph.name}'")
            lines.append(f"{indent_str}  Subgraph Definition:")
            self._dump_node_recursive(node.subgraph.root_node, lines, indent + 2, visited)

        # Children
        if node.children:
            lines.append(f"{indent_str}  Children: ({len(node.children)})")
            for child in node.children:
                self._dump_node_recursive(child, lines, indent + 2, visited)

    def get_fused_function(self, module: "Module", name: str) -> FusedFunction:
        """
        Create a FusedFunction instance for this fuser.

        Args:
            module: Module to associate with the fused function
            name: Name for the fused function

        Returns:
            A FusedFunction instance
        """
        return FusedFunction(self, module, name)

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
                node.inputs[i].type = param.type

            # Get return type
            return_type = node.function._slang_func.return_type
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
        input_params = ", ".join(
            f"{port.type.full_name if port.type else 'auto'} {port.name}" for port in node.inputs
        )
        return_type = (
            node.outputs[0].type.full_name if node.outputs and node.outputs[0].type else "void"
        )
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
            sorted_children = _sort_graph(node)

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
                    output_type = (
                        child.outputs[0].type.full_name if child.outputs[0].type else "auto"
                    )
                    body_lines.append(
                        f"\t{output_type} {temp_var} = {child.name}({', '.join(child_args)});"
                    )

            # Generate return statement
            for output_port in node.outputs:
                if output_port.source is not None:
                    source_node, source_output_name = output_port.source
                    if source_node is not None:
                        # Output comes from a child
                        temp_var = child_output_to_temp.get((source_node, source_output_name))
                        if temp_var:
                            body_lines.append(f"\treturn {temp_var};")
                        else:
                            body_lines.append(f"\treturn <missing:{source_output_name}>;")
                    else:
                        # Output comes directly from an input
                        body_lines.append(f"\treturn {source_output_name};")
        else:
            # Leaf node with no children
            if node.function is not None:
                # Node has an associated function - generate a call to it
                func_args = ", ".join(port.name for port in node.inputs)
                body_lines.append(f"\treturn {node.function.name}({func_args});")
            else:
                body_lines.append("\treturn <not worked out yet>")

        return "\n".join(body_lines)
