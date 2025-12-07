# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Bytecode-based fusion system for SlangPy.

This module implements a VM-based approach to function fusion where operations
are stored as a sequence of bytecode instructions that operate on symbolic variables.
"""

from typing import TYPE_CHECKING, Optional, Union
from enum import Enum, auto
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from slangpy.core.function import Function
    from slangpy.reflection.reflectiontypes import SlangType, SlangFunction
    from slangpy.bindings.marshall import BindContext
    from slangpy.bindings.boundvariable import BoundVariable
    from slangpy.core.native import NativeMarshall


class OpCode(Enum):
    """Bytecode operation codes for the fusion VM."""

    CALL_SLANG = auto()  # Call a Slang function
    CALL_SUB = auto()  # Call a sub-function (nested VM)


@dataclass
class Variable:
    """
    Represents a symbolic variable in the VM.

    Variables are allocated sequentially and can represent:
    - Input parameters to the fused function
    - Intermediate results from operations
    - Output values
    """

    id: int  # Unique sequential ID
    name: str  # Human-readable name for debugging
    slang: Optional["SlangType"] = None  # Slang type (None if not yet inferred)
    python: Optional["NativeMarshall"] = None  # Python marshal for type resolution

    def __str__(self) -> str:
        type_str = self.slang.full_name if self.slang else "unknown"
        return f"v{self.id}:{self.name}:{type_str}"

    def __repr__(self) -> str:
        return f"Variable(id={self.id}, name='{self.name}', slang={self.slang})"


@dataclass
class Instruction:
    """
    Base class for bytecode instructions.

    Each instruction operates on variables and may produce output variables.
    """

    opcode: OpCode
    inputs: list[int]  # Variable IDs used as input
    outputs: list[int]  # Variable IDs produced as output

    def __str__(self) -> str:
        inputs_str = ", ".join(f"v{i}" for i in self.inputs)
        outputs_str = ", ".join(f"v{i}" for i in self.outputs)
        return f"{self.opcode.name}({inputs_str}) -> ({outputs_str})"


class CallSlangInstruction(Instruction):
    """
    Instruction to call a Slang function.

    Attributes:
        function: The Slang function to call
        inputs: Variable IDs to pass as arguments
        outputs: Variable IDs to store results (typically one for return value)
    """

    def __init__(self, function: Optional["Function"], inputs: list[int], outputs: list[int]):
        super().__init__(OpCode.CALL_SLANG, inputs, outputs)
        self.function = function

    def __str__(self) -> str:
        func_name = self.function.name if self.function else "unknown"
        inputs_str = ", ".join(f"v{i}" for i in self.inputs)
        outputs_str = ", ".join(f"v{i}" for i in self.outputs)
        return f"CALL_SLANG {func_name}({inputs_str}) -> ({outputs_str})"


class CallSubInstruction(Instruction):
    """
    Instruction to call a sub-function (nested VM).

    Attributes:
        sub_program: The nested FuseProgram to execute
        inputs: Variable IDs to pass as arguments
        outputs: Variable IDs to store results
    """

    def __init__(self, sub_program: Optional["FuseProgram"], inputs: list[int], outputs: list[int]):
        super().__init__(OpCode.CALL_SUB, inputs, outputs)
        self.sub_program = sub_program

    def __str__(self) -> str:
        sub_name = self.sub_program.name if self.sub_program else "unknown"
        inputs_str = ", ".join(f"v{i}" for i in self.inputs)
        outputs_str = ", ".join(f"v{i}" for i in self.outputs)
        return f"CALL_SUB {sub_name}({inputs_str}) -> ({outputs_str})"


class FuseProgram:
    """
    A bytecode program representing a fused function.

    The program consists of:
    - A sequence of instructions
    - A pool of symbolic variables
    - Metadata about inputs and outputs
    """

    def __init__(self, name: str):
        """
        Initialize a new fusion program.

        Args:
            name: Name of the fused function
        """
        self.name = name
        self.instructions: list[Instruction] = []
        self.variables: list[Variable] = []
        self._next_var_id = 0

        # Track which variables are inputs/outputs
        self.input_vars: list[int] = []  # Variable IDs that are function inputs
        self.output_vars: list[int] = []  # Variable IDs that are function outputs

    def allocate_variable(self, name: str, slang: Optional["SlangType"] = None) -> int:
        """
        Allocate a new variable in the program.

        Args:
            name: Human-readable name for debugging
            slang: Optional Slang type (can be inferred later)

        Returns:
            Variable ID (index in variables list)
        """
        var_id = self._next_var_id
        self._next_var_id += 1
        var = Variable(id=var_id, name=name, slang=slang)
        self.variables.append(var)
        return var_id

    def get_variable(self, var_id: int) -> Variable:
        """
        Get a variable by its ID.

        Args:
            var_id: Variable ID

        Returns:
            The Variable object

        Raises:
            IndexError: If variable ID is invalid
        """
        if var_id < 0 or var_id >= len(self.variables):
            raise IndexError(f"Invalid variable ID: {var_id}")
        return self.variables[var_id]

    def add_instruction(self, instruction: Instruction) -> None:
        """
        Add an instruction to the program.

        Args:
            instruction: The instruction to add
        """
        self.instructions.append(instruction)

    def add_input(self, var_id: int) -> None:
        """
        Mark a variable as a function input parameter.

        Args:
            var_id: Variable ID
        """
        if var_id not in self.input_vars:
            self.input_vars.append(var_id)

    def add_output(self, var_id: int) -> None:
        """
        Mark a variable as a function output.

        Args:
            var_id: Variable ID
        """
        if var_id not in self.output_vars:
            self.output_vars.append(var_id)

    def __str__(self) -> str:
        """Generate a human-readable string representation."""
        lines = []
        lines.append(f"FuseProgram: {self.name}")
        lines.append("=" * 60)

        # Show inputs
        if self.input_vars:
            inputs_str = ", ".join(str(self.get_variable(v)) for v in self.input_vars)
            lines.append(f"Inputs: {inputs_str}")

        # Show outputs
        if self.output_vars:
            outputs_str = ", ".join(str(self.get_variable(v)) for v in self.output_vars)
            lines.append(f"Outputs: {outputs_str}")

        lines.append("")
        lines.append("Variables:")
        for var in self.variables:
            lines.append(f"  {var}")

        lines.append("")
        lines.append("Instructions:")
        for i, instr in enumerate(self.instructions):
            lines.append(f"  {i:3d}: {instr}")

        return "\n".join(lines)

    def dump(self) -> str:
        """
        Generate a detailed dump of the program for debugging.

        Returns:
            Multi-line string with full program details
        """
        return str(self)

    def infer_types(
        self,
        bind_context: Optional["BindContext"] = None,
        input_marshals: Optional[
            list[Union["NativeMarshall", "SlangType", "BoundVariable"]]
        ] = None,
    ) -> bool:
        """
        Infer types for all variables in the program by propagating type information
        through instructions.

        This assumes that:
        - Input variables may or may not have types already set
        - Slang functions have known parameter and return types
        - Types flow from inputs through instructions to outputs

        Args:
            bind_context: Optional BindContext for proper type resolution
            input_marshals: Optional list of NativeMarshall, SlangType, or BoundVariable for input variables (matched by index)
                           If BoundVariable is provided, its .python property (NativeMarshall) will be extracted.

        Returns:
            True if all variable types were successfully inferred, False otherwise
        """
        # Associate input marshals with input variables
        if input_marshals is not None:
            if len(input_marshals) != len(self.input_vars):
                raise ValueError(
                    f"Number of input marshals ({len(input_marshals)}) does not match "
                    f"number of input variables ({len(self.input_vars)})"
                )
            # Store marshals on variables, extracting from BoundVariable if needed
            for i, var_id in enumerate(self.input_vars):
                var = self.get_variable(var_id)
                marshal_or_type = input_marshals[i]

                # Extract marshal from BoundVariable if needed
                from slangpy.bindings.boundvariable import BoundVariable
                from slangpy.reflection.reflectiontypes import SlangType

                if isinstance(marshal_or_type, BoundVariable):
                    marshal_or_type = marshal_or_type.python

                # Store python marshal or extract from SlangType
                if isinstance(marshal_or_type, SlangType):
                    # If it's a SlangType, set it directly and don't store a marshal
                    var.slang = marshal_or_type
                    var.python = None
                else:
                    # It's a NativeMarshall
                    var.python = marshal_or_type

        # Iterate through instructions in order
        for instr in self.instructions:
            if isinstance(instr, CallSlangInstruction):
                self._infer_types_call_slang(instr, bind_context)
            elif isinstance(instr, CallSubInstruction):
                self._infer_types_call_sub(instr, bind_context)

        # Check if all variables have types
        return all(var.slang is not None for var in self.variables)

    def _infer_types_call_slang(
        self, instr: CallSlangInstruction, bind_context: Optional["BindContext"] = None
    ) -> None:
        """
        Infer types for a CALL_SLANG instruction.

        Requires a bind_context and python marshals on all input variables for proper type resolution.

        Args:
            instr: The CALL_SLANG instruction
            bind_context: BindContext for proper type resolution (required)

        Raises:
            ValueError: If bind_context is None or input variables lack python marshals
        """
        if instr.function is None:
            return

        func = instr.function

        # Get the underlying SlangFunction
        slang_func = func._slang_func if hasattr(func, "_slang_func") else None
        if slang_func is None:
            return

        # Require bind_context for proper type resolution
        if bind_context is None:
            raise ValueError(
                f"Cannot infer types for instruction {instr}: bind_context is required for type resolution"
            )

        # Verify all input variables have either python marshals OR slang types
        missing_info = [
            var_id
            for var_id in instr.inputs
            if self.get_variable(var_id).python is None and self.get_variable(var_id).slang is None
        ]
        if missing_info:
            var_names = [self.get_variable(vid).name for vid in missing_info]
            raise ValueError(
                f"Cannot infer types for instruction {instr}: "
                f"input variables {var_names} lack both python marshals and slang types. "
                f"Ensure all input variables have either marshals or types set before calling infer_types()."
            )

        # Use proper type resolution
        self._infer_types_with_resolution(instr, slang_func, bind_context)

    def _infer_types_with_resolution(
        self, instr: CallSlangInstruction, slang_func: "SlangFunction", bind_context: "BindContext"
    ) -> None:
        """
        Type inference using proper type resolution through the new resolve_function_call API.

        Args:
            instr: The CALL_SLANG instruction
            slang_func: The SlangFunction to resolve against
            bind_context: The BindContext for resolution
        """
        from slangpy.reflection.typeresolution import (
            resolve_function_call,
            ResolutionDiagnostic,
        )

        # Extract marshals or slang types from input variables
        # resolve_function_call accepts either NativeMarshall OR SlangType
        args = []
        for var_id in instr.inputs:
            var = self.get_variable(var_id)
            # Prefer already-resolved slang type, fallback to python marshal
            if var.slang is not None:
                args.append(var.slang)
            else:
                args.append(var.python)

        # Create diagnostics
        diagnostics = ResolutionDiagnostic()

        # Call the new type resolution API directly
        result = resolve_function_call(
            bind_context,
            slang_func,
            args,
            {},  # No kwargs for now
            diagnostics,
            None,  # No this_type
        )

        if result is None:
            # Resolution failed - this should not happen with valid inputs
            diag_str = "\n".join(diagnostics.summary_lines)
            raise ValueError(
                f"Type resolution failed for function {slang_func.name} in instruction {instr}.\n"
                f"Diagnostics:\n{diag_str}"
            )

        # Apply resolved types to input variables
        for i, var_id in enumerate(instr.inputs):
            if i < len(result.params):
                var = self.get_variable(var_id)
                var.slang = result.params[i].type

        # Apply return type to output variable
        if len(instr.outputs) > 0 and result.function.return_type is not None:
            output_var = self.get_variable(instr.outputs[0])
            output_var.slang = result.function.return_type

    def _infer_types_call_sub(
        self, instr: CallSubInstruction, bind_context: Optional["BindContext"] = None
    ) -> None:
        """
        Infer types for a CALL_SUB instruction.

        For sub-programs:
        - Input variables should match the sub-program's input types
        - Output variables should match the sub-program's output types

        Args:
            instr: The CALL_SUB instruction
            bind_context: Optional BindContext (currently unused for sub-programs)
        """
        if instr.sub_program is None:
            return

        sub_prog = instr.sub_program

        # Propagate python marshals or slang types to sub-program inputs
        # Similar to _infer_types_with_resolution, we prefer slang types if available
        if bind_context is not None:
            sub_input_types = []
            for var_id in instr.inputs:
                var = self.get_variable(var_id)
                # Prefer already-resolved slang type, fallback to python marshal
                if var.slang is not None:
                    sub_input_types.append(var.slang)
                elif var.python is not None:
                    sub_input_types.append(var.python)
                else:
                    # Variable has neither type nor marshal - can't propagate
                    sub_input_types = []
                    break

            if len(sub_input_types) == len(instr.inputs):
                # Run type inference on sub-program with these types/marshals
                sub_prog.infer_types(bind_context, sub_input_types)

        # Infer types for input variables from sub-program inputs
        for i, var_id in enumerate(instr.inputs):
            if i < len(sub_prog.input_vars):
                sub_input_var = sub_prog.get_variable(sub_prog.input_vars[i])
                var = self.get_variable(var_id)
                if var.slang is None and sub_input_var.slang is not None:
                    var.slang = sub_input_var.slang

        # Infer types for output variables from sub-program outputs
        for i, var_id in enumerate(instr.outputs):
            if i < len(sub_prog.output_vars):
                sub_output_var = sub_prog.get_variable(sub_prog.output_vars[i])
                var = self.get_variable(var_id)
                if var.slang is None and sub_output_var.slang is not None:
                    var.slang = sub_output_var.slang

    def clear_types(self) -> None:
        """
        Clear all type information from variables.
        Useful for testing or re-running type inference.
        """
        for var in self.variables:
            var.slang = None

    def generate_code(self, function_name: Optional[str] = None) -> str:
        """
        Generate Slang code from the bytecode program.

        This requires that type inference has been run first.

        Args:
            function_name: Optional name for the generated function (defaults to program name)

        Returns:
            Generated Slang code as a string

        Raises:
            ValueError: If type inference hasn't been run or incomplete
        """
        # Verify all variables have types
        if not all(var.slang is not None for var in self.variables):
            raise ValueError(
                "Cannot generate code: not all variables have types. Run infer_types() first."
            )

        if function_name is None:
            function_name = f"__func_{self.name}"

        lines = []

        # Generate function signature
        return_type = "void"
        if len(self.output_vars) > 0:
            # Use the type of the first output variable as return type
            output_var = self.get_variable(self.output_vars[0])
            assert output_var.slang is not None, "Output variable must have a slang type"
            return_type = output_var.slang.full_name

        # Generate parameter list
        params = []
        for var_id in self.input_vars:
            var = self.get_variable(var_id)
            assert var.slang is not None, f"Input variable {var.name} must have a slang type"
            params.append(f"{var.slang.full_name} v{var.id}")

        params_str = ", ".join(params)
        lines.append(f"{return_type} {function_name}({params_str})")
        lines.append("{")

        # Declare temporary and output variables
        temp_and_output_vars = []
        for var in self.variables:
            if var.id not in self.input_vars:
                temp_and_output_vars.append(var)

        if temp_and_output_vars:
            # Group by type for more compact declaration
            type_groups: dict[str, list[int]] = {}
            for var in temp_and_output_vars:
                assert var.slang is not None, f"Variable {var.name} must have a slang type"
                type_name = var.slang.full_name
                if type_name not in type_groups:
                    type_groups[type_name] = []
                type_groups[type_name].append(var.id)

            for type_name, var_ids in type_groups.items():
                var_names = ",".join(f"v{vid}" for vid in var_ids)
                lines.append(f"\t{type_name} {var_names};")

        # Generate instruction code
        for instr in self.instructions:
            if isinstance(instr, CallSlangInstruction):
                code = self._generate_code_call_slang(instr)
                lines.append(f"\t{code}")
            elif isinstance(instr, CallSubInstruction):
                code = self._generate_code_call_sub(instr)
                lines.append(f"\t{code}")

        # Generate return statement
        if len(self.output_vars) > 0:
            return_var_id = self.output_vars[0]
            lines.append(f"\treturn v{return_var_id};")

        lines.append("}")

        return "\n".join(lines)

    def _generate_code_call_slang(self, instr: CallSlangInstruction) -> str:
        """
        Generate code for a CALL_SLANG instruction.

        Args:
            instr: The instruction

        Returns:
            Generated code line
        """
        if instr.function is None:
            return "// ERROR: No function"

        # Get function name
        func = instr.function
        slang_func = func._slang_func if hasattr(func, "_slang_func") else None
        if slang_func is None:
            return "// ERROR: No slang function"

        func_name = slang_func.name

        # Generate argument list
        args = ", ".join(f"v{vid}" for vid in instr.inputs)

        # Generate assignment
        if len(instr.outputs) > 0:
            output_var_id = instr.outputs[0]
            return f"v{output_var_id} = {func_name}({args});"
        else:
            return f"{func_name}({args});"

    def _generate_code_call_sub(self, instr: CallSubInstruction) -> str:
        """
        Generate code for a CALL_SUB instruction.

        For now, this generates a call to the sub-program's generated function.
        In the future, this could inline the sub-program code.

        Args:
            instr: The instruction

        Returns:
            Generated code line
        """
        if instr.sub_program is None:
            return "// ERROR: No sub-program"

        sub_prog = instr.sub_program
        func_name = f"__func_{sub_prog.name}"

        # Generate argument list
        args = ", ".join(f"v{vid}" for vid in instr.inputs)

        # Generate assignment
        if len(instr.outputs) > 0:
            output_var_id = instr.outputs[0]
            return f"v{output_var_id} = {func_name}({args});"
        else:
            return f"{func_name}({args});"


class FuseProgramBuilder:
    """
    Helper class to build FuseProgram instances.

    Provides a fluent interface for constructing programs.
    """

    def __init__(self, name: str):
        """
        Initialize a new program builder.

        Args:
            name: Name of the fused function
        """
        self.program = FuseProgram(name)

    def input(self, name: str, type: Optional["SlangType"] = None) -> int:
        """
        Add an input variable.

        Args:
            name: Variable name
            type: Optional type

        Returns:
            Variable ID
        """
        var_id = self.program.allocate_variable(name, type)
        self.program.add_input(var_id)
        return var_id

    def temp(self, name: str, type: Optional["SlangType"] = None) -> int:
        """
        Add a temporary variable.

        Args:
            name: Variable name
            type: Optional type

        Returns:
            Variable ID
        """
        return self.program.allocate_variable(name, type)

    def output(self, name: str, type: Optional["SlangType"] = None) -> int:
        """
        Add an output variable.

        Args:
            name: Variable name
            type: Optional type

        Returns:
            Variable ID
        """
        var_id = self.program.allocate_variable(name, type)
        self.program.add_output(var_id)
        return var_id

    def call_slang(
        self, function: "Function", inputs: list[int], outputs: list[int]
    ) -> "FuseProgramBuilder":
        """
        Add a CALL_SLANG instruction.

        Args:
            function: Slang function to call
            inputs: Input variable IDs
            outputs: Output variable IDs

        Returns:
            Self for chaining
        """
        instr = CallSlangInstruction(function, inputs, outputs)
        self.program.add_instruction(instr)
        return self

    def call_sub(
        self, sub_program: "FuseProgram", inputs: list[int], outputs: list[int]
    ) -> "FuseProgramBuilder":
        """
        Add a CALL_SUB instruction.

        Args:
            sub_program: Sub-program to call
            inputs: Input variable IDs
            outputs: Output variable IDs

        Returns:
            Self for chaining
        """
        instr = CallSubInstruction(sub_program, inputs, outputs)
        self.program.add_instruction(instr)
        return self

    def build(self) -> FuseProgram:
        """
        Get the built program.

        Returns:
            The constructed FuseProgram
        """
        return self.program
