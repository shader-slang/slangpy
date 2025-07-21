# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, Optional, Union, cast

from slangpy.core.enums import IOType
from slangpy.core.native import AccessType, CallMode, Shape, NativeMarshall

from slangpy import ModifierID
from slangpy.bindings.marshall import BindContext
from slangpy.bindings.codegen import CodeGen
from slangpy.bindings.typeregistry import get_or_create_type
from slangpy.reflection import SlangField, SlangFunction, SlangParameter, SlangType


class BoundVariableException(Exception):
    """
    Custom exception type that carries a message and the variable that caused
    the exception.
    """

    def __init__(self, message: str, variable: "BoundVariable"):
        super().__init__(message)
        self.message = message
        self.variable = variable


class BoundCall:
    """
    Stores the binding of python arguments to slang parameters during kernel
    generation. This is initialized purely with a set of python arguments and
    later bound to corresponding slang parameters during function resolution.
    """

    def __init__(self, context: "BindContext", *args: Any, **kwargs: Any):
        super().__init__()
        self.args = [BoundVariable(context, x, None, "", i) for (i, x) in enumerate(args)]
        self.kwargs = {n: BoundVariable(context, v, None, n) for n, v in kwargs.items()}

    def bind(self, slang: SlangFunction):
        """
        Stores slang function this call is bound to.
        """
        self.slang = slang

    @property
    def differentiable(self) -> bool:
        """
        Returns whether this call is differentiable.
        """
        return self.slang.differentiable

    @property
    def num_function_args(self) -> int:
        """
        Returns total arguments passed to the function, excluding
        special values such as _this and _result.
        """
        total = len(self.args) + self.num_function_kwargs
        return total

    @property
    def num_function_kwargs(self) -> int:
        """
        Returns total keyword arguments passed to the function, excluding
        special values such as _this and _result.
        """
        total = len(self.kwargs)
        if "_this" in self.kwargs:
            total -= 1
        if "_result" in self.kwargs:
            total -= 1
        return total

    @property
    def has_implicit_args(self) -> bool:
        """
        Returns whether any arguments need their types resolving
        implicitly.
        """
        return any(x.vector_type is None for x in self.args)

    @property
    def has_implicit_mappings(self) -> bool:
        """
        Returns whether any arguments need their mappings resolving
        implicitly.
        """
        return any(not x.vector_mapping.valid for x in self.args)

    def apply_explicit_vectorization(
        self, context: "BindContext", args: tuple[Any, ...], kwargs: dict[str, Any]
    ):
        """
        Calls apply_explicit_vectorization on all arguments, which calculates
        the vector type and mapping for each argument based on explicitly
        provided information via function.map().
        """

        if len(args) > len(self.args):
            raise ValueError("Too many arguments supplied for explicit vectorization")
        if len(kwargs) > len(self.kwargs):
            raise ValueError("Too many keyword arguments supplied for explicit vectorization")

        for i, arg in enumerate(args):
            self.args[i].apply_explicit_vectorization(context, arg)

        for name, arg in kwargs.items():
            if not name in self.kwargs:
                raise ValueError(f"Unknown keyword argument {name}")
            self.kwargs[name].apply_explicit_vectorization(context, arg)

    def values(self) -> list["BoundVariable"]:
        """
        Return list of all bound variables in the call.
        """
        return self.args + list(self.kwargs.values())

    def apply_implicit_vectorization(self, context: BindContext):
        """
        Calls apply_implicit_vectorization on all arguments, which attempts
        to calculate any remaining vector types once binding to
        a slang function is complete.
        """

        for arg in self.args:
            arg.apply_implicit_vectorization(context)

        for arg in self.kwargs.values():
            arg.apply_implicit_vectorization(context)

    def finalize_mappings(self, context: BindContext):
        """
        Calls finalize_mappings on all arguments, which ensures all vector
        mappings are valid and consistent.
        """
        for arg in self.args:
            arg.finalize_mappings(context)

        for arg in self.kwargs.values():
            arg.finalize_mappings(context)


class BoundVariable:
    """
    Node in a built signature tree, maintains a pairing of python+slang marshall,
    and a potential set of child nodes for use during kernel generation.
    """

    def __init__(
        self,
        context: "BindContext",
        value: Any,
        parent: Optional["BoundVariable"],
        name: str,
        python_pos_arg_index: int = -1,
    ):

        super().__init__()

        # Store the python and slang marshall
        #: The name of the variable
        self.name = name

        #: Index in python positional arguments (used for debug information)
        self.python_pos_arg_index = python_pos_arg_index

        #: The name of the variable in the generated code
        self.variable_name = name

        #: Access type for primal and derivative
        self.access = (AccessType.none, AccessType.none)

        #: Is this variable differentiable
        self.differentiable = False

        #: Call dimensionality of this variable.
        self.call_dimensionality = None

        #: Parameter index in slang function parameter list.
        self.param_index = -1

        #: Mapping of variable dimensions to call dimensions.
        self.vector_mapping: Shape = Shape(None)

        #: Vector type variable will mapped to.
        self.vector_type: Optional[SlangType] = None

        #: Whether this type had vectorization explicitly specified
        self.explicitly_vectorized = False

        #: Slang type this variable is bound to.
        self.slang_type: Optional[SlangType] = None

        # Initialize path
        if parent is None:
            # Path relative to root.
            self.path = self.name
        else:
            self.path = f"{parent.path}.{self.name}"

        #: The python marshall for this variable
        self.python: NativeMarshall

        import slangpy.core.packedarg

        if isinstance(value, slangpy.core.packedarg.PackedArg):
            # If this is a packed arg, we need to use the slang type from the marshall
            if parent is not None:
                raise BoundVariableException(
                    f"PackedArg {value} must be a top level argument", self
                )
            self.python = value.python
            self.create_param_block = True
        else:
            # Not packed arg so we need to create a marshall for the value
            try:
                self.python = get_or_create_type(context.layout, type(value), value)
            except Exception as e:
                raise BoundVariableException(
                    f"Failed to create type marshall for argument {self.debug_name}: {value} with error {e}",
                    self,
                ) from e
            self.create_param_block = False

        # Create children
        # TODO: Should this be based off type fields
        if isinstance(value, dict):
            # Child variables.
            self.children = {
                n: BoundVariable(context, v, self, n) for n, v in value.items() if n != "_type"
            }
        else:
            self.children = None

    @property
    def debug_name(self) -> str:
        if self.path != "":
            return self.path
        elif self.name != "":
            return self.name
        else:
            return f"arg{self.python_pos_arg_index}"

    def bind(
        self,
        slang: Union[SlangField, SlangParameter, SlangType],
        modifiers: set[ModifierID] = set(),
        override_name: Optional[str] = None,
    ):
        """
        Bind to a given slang field, parameter or type. Stores the slang type and modifiers,
        and recursively binds children.
        """
        if isinstance(slang, SlangType):
            if self.name == "":
                assert override_name is not None
                self.name = override_name
            else:
                self.name = self.name
            self.slang_type = slang
            self.slang_modifiers = modifiers
        else:
            self.name = slang.name
            self.slang_type = slang.type
            self.slang_modifiers = modifiers.union(slang.modifiers)
        self.variable_name = self.name

        if self.children is not None:
            for child in self.children.values():
                if child.name not in self.slang_type.fields:
                    raise ValueError(
                        f"Slang type '{self.slang_type.full_name}' has no field '{child.name}'"
                    )
                slang_child = self.slang_type.fields[child.name]
                child.bind(slang_child, self.slang_modifiers)

    @property
    def io_type(self) -> IOType:
        """
        Returns the IO type of this variable based on slang modifiers.
        """

        have_in = ModifierID.inn in self.slang_modifiers
        have_out = ModifierID.out in self.slang_modifiers
        have_inout = ModifierID.inout in self.slang_modifiers

        if (have_in and have_out) or have_inout:
            return IOType.inout
        elif have_out:
            return IOType.out
        else:
            return IOType.inn

    @property
    def no_diff(self) -> bool:
        """
        Returns whether this variable is marked as no_diff.
        """
        return ModifierID.nodiff in self.slang_modifiers

    def apply_explicit_vectorization(self, context: "BindContext", mapping: Any):
        """
        Apply explicit vectorization to this variable and children.
        This will result in any explicit mapping or typing provided
        by the caller being stored on the corresponding bound variable.
        """
        if self.children is not None:

            if isinstance(mapping, dict):
                for name, child in self.children.items():
                    child_mapping = mapping.get(name)
                    if child_mapping is not None:
                        assert isinstance(child, BoundVariable)
                        child.apply_explicit_vectorization(context, child_mapping)

                type_mapping = mapping.get("$type")
                if type_mapping is not None:
                    self._apply_explicit_vectorization(context, type_mapping)
            else:
                self._apply_explicit_vectorization(context, mapping)
        else:
            self._apply_explicit_vectorization(context, mapping)

    def _apply_explicit_vectorization(self, context: "BindContext", mapping: Any):
        from slangpy.core.struct import Struct

        try:
            if isinstance(mapping, tuple):
                self.vector_mapping = Shape(*mapping)
                self.vector_type = cast(SlangType, self.python.reduce_type(context, len(mapping)))
                self.explicitly_vectorized = True
            elif isinstance(mapping, SlangType):
                self.vector_type = mapping
                self.explicitly_vectorized = True
            elif isinstance(mapping, Struct):
                self.vector_type = mapping.struct
                self.explicitly_vectorized = True
            elif isinstance(mapping, str):
                self.vector_type = context.layout.find_type_by_name(mapping)
                self.explicitly_vectorized = True
            elif isinstance(mapping, type):
                marshall = get_or_create_type(context.layout, mapping)
                if not marshall:
                    raise BoundVariableException(f"Invalid explicit type: {mapping}", self)
                self.vector_type = cast(SlangType, marshall.slang_type)
                self.explicitly_vectorized = True
            else:
                raise BoundVariableException(f"Invalid explicit type: {mapping}", self)
        except Exception as e:
            raise BoundVariableException(
                f"Explicit vectorization raised exception: {e.__repr__()}", self
            )

    def apply_implicit_vectorization(self, context: BindContext):
        """
        Apply implicit vectorization to this variable. This inspects
        the slang type being bound to in an attempt to get a concrete
        type once the slang function is known.
        """
        if self.children is not None:
            for child in self.children.values():
                child.apply_implicit_vectorization(context)
        self._apply_implicit_vectorization(context)

    def _apply_implicit_vectorization(self, context: BindContext):
        if self.vector_mapping.valid:
            # if we have a valid vector mapping, just need to reduce it
            self.vector_type = cast(
                SlangType, self.python.reduce_type(context, len(self.vector_mapping))
            )

        if self.vector_type is not None:
            # do nothing in first phase if already have a type. vector
            # mapping will be worked out once specialized slang function is known
            pass
        elif self.path == "_result":
            # result is inferred last
            pass
        else:
            # neither specified, attempt to resolve type
            assert self.slang_type is not None
            self.vector_type = cast(SlangType, self.python.resolve_type(context, self.slang_type))

        # If we ended up with no valid type, use slang type. Currently this should
        # only happen for auto-allocated result buffers
        if not self.vector_mapping.valid and self.vector_type is None:
            assert self.path == "_result"
            self.vector_type = self.slang_type

        # Clear slang type info - it should never be used after this
        # Note: useful for debugging so keeping for now!
        self.slang_type = None

        # Can now calculate dimensionality
        if self.vector_mapping.valid:
            if len(self.vector_mapping) > 0:
                self.call_dimensionality = max(self.vector_mapping.as_tuple()) + 1
            else:
                self.call_dimensionality = 0
        elif self.python.match_call_shape:
            self.call_dimensionality = -1
        else:
            assert self.vector_type is not None
            self.call_dimensionality = self.python.resolve_dimensionality(
                context, self, self.vector_type
            )
            if self.call_dimensionality is not None and self.call_dimensionality < 0:
                raise BoundVariableException(
                    f"Could not resolve dimensionality for {self.path}", self
                )

    def finalize_mappings(self, context: BindContext):
        """
        Finalize vector mappings and types for this variable and children.
        """
        if self.children is not None:
            for child in self.children.values():
                child.finalize_mappings(context)
        self._finalize_mappings(context)

    def _finalize_mappings(self, context: BindContext):
        if self.call_dimensionality == -1:
            self.call_dimensionality = context.call_dimensionality

        if (
            context.options["strict_broadcasting"]
            and self.children is None
            and not self.explicitly_vectorized
        ):
            if (
                self.call_dimensionality != 0
                and self.call_dimensionality != context.call_dimensionality
            ):
                raise BoundVariableException(
                    f"""
Strict broadcasting is enabled and the argument `{self.name}` at parameter index `{self.param_index}` has dimensionality of ({self.call_dimensionality})
while the kernel's dimensionality is ({context.call_dimensionality}).
Note: This error usually happens when the arguments in a call have different dimensionalities. By default SlangPy will automatically broadcast
the dimentionality of arguments, however if 'strict_broadcasting' option is enabled when loading the module, SlangPy will not perform the broadcasting.
(See: https://slangpy.shader-slang.org/en/latest/src/basics/broadcasting.html for details).
You can disable strict broadcasting by not setting the `strict_broadcasting` option when loading the module.
More Advancedly, SlangPy provides a `mapping` functionality to finer control the argument's dimensionality,
you can find more information in the Mapping section of the documentation (https://slangpy.shader-slang.org/en/latest/src/basics/mapping.html).
""",
                    self,
                )

        if not self.vector_mapping.valid:
            assert self.call_dimensionality is not None
            m: list[int] = []
            for i in range(self.call_dimensionality):
                m.append(context.call_dimensionality - i - 1)
            m.reverse()
            self.vector_mapping = Shape(*m)

    def calculate_differentiability(self, context: BindContext):
        """
        Recursively calculate  differentiability
        """

        # Can now decide if differentiable
        assert self.vector_type is not None
        self.differentiable = (
            not self.no_diff and self.vector_type.differentiable and self.python.has_derivative
        )
        self._calculate_differentiability(context.call_mode)

        if self.children is not None:
            for child in self.children.values():
                child.calculate_differentiability(context)

    def get_input_list(self, args: list["BoundVariable"]):
        """
        Recursively populate flat list of argument nodes
        """
        self._get_input_list_recurse(args)
        return args

    def _get_input_list_recurse(self, args: list["BoundVariable"]):
        """
        Internal recursive function to populate flat list of argument nodes
        """
        if self.children is not None:
            for child in self.children.values():
                child._get_input_list_recurse(args)
        else:
            args.append(self)

    def __repr__(self):
        return self.python.__repr__()

    def _calculate_differentiability(self, mode: CallMode):
        """
        Calculates access types based on differentiability, call mode and io type
        """
        if mode == CallMode.prim:
            if self.differentiable:
                if self.io_type == IOType.inout:
                    self.access = (AccessType.readwrite, AccessType.none)
                elif self.io_type == IOType.out:
                    self.access = (AccessType.write, AccessType.none)
                else:
                    self.access = (AccessType.read, AccessType.none)
            else:
                if self.io_type == IOType.inout:
                    self.access = (AccessType.readwrite, AccessType.none)
                elif self.io_type == IOType.out:
                    self.access = (AccessType.write, AccessType.none)
                else:
                    self.access = (AccessType.read, AccessType.none)
        elif mode == CallMode.bwds:
            if self.differentiable:
                if self.io_type == IOType.inout:
                    self.access = (AccessType.read, AccessType.readwrite)
                elif self.io_type == IOType.out:
                    self.access = (AccessType.none, AccessType.read)
                else:
                    self.access = (AccessType.read, AccessType.write)
            else:
                if self.io_type == IOType.inout:
                    self.access = (AccessType.read, AccessType.none)
                elif self.io_type == IOType.out:
                    self.access = (AccessType.none, AccessType.none)
                else:
                    self.access = (AccessType.read, AccessType.none)
        else:
            # todo: fwds
            self.access = (AccessType.none, AccessType.none)

    def gen_call_data_code(self, cg: CodeGen, context: BindContext, depth: int = 0):
        if self.children is not None:
            cgb = cg.call_data_structs

            cgb.begin_struct(f"_t_{self.variable_name}")

            for field, variable in self.children.items():
                variable.gen_call_data_code(cg, context, depth + 1)

            for var in self.children.values():
                cgb.declare(f"_t_{var.variable_name}", var.variable_name)

            assert self.vector_type is not None
            context_decl = f"ContextND<{self.call_dimensionality}> context"
            value_decl = f"{self.vector_type.full_name} value"
            prefix = "[Differentiable]" if self.access[1] != AccessType.none else ""

            cgb.empty_line()
            cgb.append_line(f"{prefix} void load({context_decl}, out {value_decl})")
            cgb.begin_block()
            for field, var in self.children.items():
                cgb.append_statement(
                    f"{var.variable_name}.load(context.map(_m_{var.variable_name}),value.{field})"
                )
            cgb.end_block()

            if self.access[0] in (AccessType.write, AccessType.readwrite):
                cgb.empty_line()
                cgb.append_line(f"{prefix} void store({context_decl}, in {value_decl})")
                cgb.begin_block()
                for field, var in self.children.items():
                    cgb.append_statement(
                        f"{var.variable_name}.store(context.map(_m_{var.variable_name}),value.{field})"
                    )
                cgb.end_block()

            cgb.end_struct()

        else:
            # Generate call data
            self.python.gen_calldata(cg.call_data_structs, context, self)

        if len(self.vector_mapping) > 0:
            cg.call_data_structs.append_statement(
                f"static const int[] _m_{self.variable_name} = {{ {','.join([str(x) for x in self.vector_mapping.as_tuple()])} }}"
            )
        else:
            cg.call_data_structs.append_statement(f"static const int _m_{self.variable_name} = 0")

        if depth == 0:
            if self.create_param_block:
                cg.add_parameter_block(f"_t_{self.variable_name}", "_param_" + self.variable_name)
            else:
                cg.call_data.declare(f"_t_{self.variable_name}", self.variable_name)

    def _gen_trampoline_argument(self):
        assert self.vector_type is not None
        arg_def = f"{self.vector_type.full_name} {self.variable_name}"
        if self.io_type == IOType.inout:
            arg_def = f"inout {arg_def}"
        elif self.io_type == IOType.out:
            arg_def = f"out {arg_def}"
        elif self.io_type == IOType.inn:
            arg_def = f"in {arg_def}"
        if self.no_diff or not self.differentiable:
            arg_def = f"no_diff {arg_def}"
        return arg_def

    def __str__(self) -> str:
        return self._recurse_str(0)

    def _recurse_str(self, depth: int) -> str:
        if self.children is not None:
            child_strs = [
                f"{'  ' * depth}{name}: {child._recurse_str(depth + 1)}"
                for name, child in self.children.items()
            ]
            return "\n" + "\n".join(child_strs)
        else:
            return f"{self.name}"
