# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Fused Function Interface

This module provides the interface layer between fused programs (fusevm.py)
and the standard SlangPy call system (calldata.py, callsignature.py).

It creates a DUC-typed wrapper around FuseProgram that allows it to be
treated as a SlangFunction for the purpose of function calls.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from slangpy.experimental.fusevm import FuseProgram
    from slangpy.reflection.reflectiontypes import SlangType
    from slangpy.bindings import BindContext


class FusedFunction:
    """
    Duck-typed wrapper around FuseProgram that mimics SlangFunction interface.

    This allows fused programs to integrate with the standard call machinery
    in calldata.py and callsignature.py.
    """

    def __init__(self, fuse_program: "FuseProgram"):
        """
        Create a FusedFunction wrapper.

        Args:
            fuse_program: The FuseProgram to wrap
        """
        self.m_fuse_program = fuse_program
        self.m_name = fuse_program.name
        self.m_is_overloaded = False
        self.m_static = True  # Fused functions are always static

    @property
    def fuse_program(self) -> "FuseProgram":
        """Get the wrapped FuseProgram."""
        return self.m_fuse_program

    @property
    def name(self) -> str:
        """Function name."""
        return self.m_name

    @property
    def full_name(self) -> str:
        """Full function name (same as name for fused functions)."""
        return self.m_name

    @property
    def is_overloaded(self) -> bool:
        """Fused functions don't support overloading."""
        return False

    @property
    def static(self) -> bool:
        """Fused functions are always static."""
        return True

    @property
    def is_constructor(self) -> bool:
        """Fused functions are not constructors."""
        return False

    @property
    def return_type(self) -> Optional["SlangType"]:
        """
        Get the return type of the fused function.
        Returns the slang type of the first output variable if available.
        """
        if len(self.m_fuse_program.output_vars) > 0:
            output_var = self.m_fuse_program.get_variable(self.m_fuse_program.output_vars[0])
            return output_var.slang
        return None

    @property
    def parameters(self) -> list:
        """
        Get function parameters.
        For fused functions, these are derived from input variables.
        """
        # Return a list of FusedParameter objects
        params = []
        for var_id in self.m_fuse_program.input_vars:
            var = self.m_fuse_program.get_variable(var_id)
            params.append(FusedParameter(var))
        return params

    @property
    def reflection(self):
        """Fake reflection object for error messages."""
        # Return None - fused functions don't have real reflection
        return None

    @property
    def differentiable(self) -> bool:
        """Fused functions are not differentiable."""
        return False

    def infer_types_from_args(
        self,
        bind_context: "BindContext",
        args: list,
        kwargs: dict,
    ) -> bool:
        """
        Run type inference on the fused program using provided arguments.

        Args:
            bind_context: The BindContext for type resolution
            args: Positional arguments (marshals or types)
            kwargs: Keyword arguments (marshals or types)

        Returns:
            True if type inference succeeded
        """
        # For now, we only support positional arguments
        if len(kwargs) > 0:
            raise ValueError("Fused functions do not yet support keyword arguments")

        # Run type inference
        return self.m_fuse_program.infer_types(bind_context, args)

    def generate_code(self, func_name: Optional[str] = None) -> str:
        """
        Generate the Slang code for this fused function.

        Args:
            func_name: Optional custom function name

        Returns:
            Generated Slang code
        """
        return self.m_fuse_program.generate_code(func_name)


class FusedParameter:
    """
    Duck-typed wrapper around a FuseProgram Variable that mimics SlangParameter.
    """

    def __init__(self, variable):
        """
        Create a FusedParameter from a Variable.

        Args:
            variable: The Variable to wrap
        """
        self.m_variable = variable

    @property
    def name(self) -> str:
        """Parameter name."""
        return self.m_variable.name

    @property
    def type(self) -> Optional["SlangType"]:
        """Parameter type (may be None before type inference)."""
        return self.m_variable.slang

    @property
    def modifiers(self):
        """Parameter modifiers (empty for fused parameters)."""
        return []
