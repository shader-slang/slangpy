from types import NoneType
from typing import Any, Optional, cast

from kernelfunctions.shapes import TShapeOrTuple

from .enums import PrimType, IOType
from .pythonvariable import PythonVariable
from .slangvariable import SlangVariable
from .codegen import CodeGen
from .native import AccessType, CallMode, Shape
from .basetype import BindContext


class BoundVariableException(Exception):
    def __init__(self, message: str, variable: 'BoundVariable') -> NoneType:
        super().__init__(message)
        self.message = message
        self.variable = variable


class BoundCall:
    def __init__(self) -> NoneType:
        super().__init__()
        self.args: list['BoundVariable'] = []
        self.kwargs: dict[str, 'BoundVariable'] = {}

    def values(self) -> list['BoundVariable']:
        return self.args + list(self.kwargs.values())


class BoundVariable:
    """
    Node in a built signature tree, maintains a pairing of python+slang marshall,
    and a potential set of child nodes
    """

    def __init__(self, python: PythonVariable, slang: SlangVariable,
                 mode: CallMode,
                 input_transforms: Optional[dict[str, TShapeOrTuple]] = None,
                 output_transforms: Optional[dict[str, TShapeOrTuple]] = None,
                 path: Optional[str] = None):

        super().__init__()

        # Store the python and slang marshall
        self.python = python
        self.slang = slang

        # Initialize path
        if path is None:
            self.path = self.slang.name
        else:
            self.path = f"{path}.{self.slang.name}"

        # Allow python info to complete any missing type info from bound slang variable
        self.python.update_from_slang_type(self.slang.primal)

        # Get the python marshall for the value + load some basic info
        self.access = (AccessType.none, AccessType.none)
        self.variable_name = self.path.replace(".", "__")

        # Can now decide if differentiable
        self.differentiable = not self.slang.no_diff and self.slang.derivative is not None and self.python.differentiable
        self._calculate_differentiability(mode)

        # Store transforms
        self.call_dimensionality = None
        self.transform: Shape = Shape(None)
        if output_transforms is not None:
            t = output_transforms.get(self.path)
            if t is not None:
                self.transform = Shape(t)

        # Create children if python value has children
        self.children: Optional[dict[str, BoundVariable]] = None
        if python.fields is not None:
            assert slang.fields is not None
            self.children = {}
            for name, child_python in python.fields.items():
                child_slang = slang.fields[name]
                self.children[name] = BoundVariable(
                    cast(PythonVariable, child_python),
                    cast(SlangVariable, child_slang),
                    mode, input_transforms, output_transforms, self.path)

    @property
    def param_index(self):
        return self.slang.param_index

    def calculate_transform(self):
        """
        Recursively calculate argument shapes for the node
        """
        if self.children is not None:
            for child in self.children.values():
                child.calculate_transform()
        else:
            self._calculate_transform()

    def get_input_list(self, args: list['BoundVariable']):
        """
        Recursively populate flat list of argument nodes
        """
        self._get_input_list_recurse(args)
        return args

    def _get_input_list_recurse(self, args: list['BoundVariable']):
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

    def _calculate_transform(self):
        # Get shape of inputs and parameter
        input_dim = self.python.dimensionality
        param_shape = self.slang.primal.get_shape()

        # Check if we have input
        if input_dim is not None:

            # If user transform was provided use it, otherwise just store a transform
            # of correct size but with all undefined values
            if self.transform.valid:
                if len(self.transform) != input_dim:
                    raise BoundVariableException(
                        f"Output transforms {self.transform} must have the same number of dimensions as the input {input_dim}", self)
            else:
                self.transform = Shape((-1,) * input_dim)

            # Dimensionality is the highest output dimension minus parameter shape
            assert self.transform is not None
            dim_count = len(self.transform)
            for x in self.transform:
                if x >= 0:
                    dim_count = max(dim_count, x+1)
            self.call_dimensionality = dim_count - len(param_shape)

            # TODO: At this point, could perform some degree of shape matching, given
            # final transform is known and potentially have concrete shape info. Would
            # need to know which dimensions were concrete vs not though.

        else:
            if self.transform.valid:
                raise BoundVariableException(
                    f"Output transforms can only be applied to variables with well defined input shape", self)
            self.transform = Shape(None)
            self.call_dimensionality = None

    def _calculate_differentiability(self, mode: CallMode):
        """
        Calculates access types based on differentiability, call mode and io type
        """
        if mode == CallMode.prim:
            if self.differentiable:
                if self.slang.io_type == IOType.inout:
                    self.access = (AccessType.readwrite, AccessType.none)
                elif self.slang.io_type == IOType.out:
                    self.access = (AccessType.write, AccessType.none)
                else:
                    self.access = (AccessType.read, AccessType.none)
            else:
                if self.slang.io_type == IOType.inout:
                    self.access = (AccessType.readwrite, AccessType.none)
                elif self.slang.io_type == IOType.out:
                    self.access = (AccessType.write, AccessType.none)
                else:
                    self.access = (AccessType.read, AccessType.none)
        elif mode == CallMode.bwds:
            if self.differentiable:
                if self.slang.io_type == IOType.inout:
                    self.access = (AccessType.read, AccessType.readwrite)
                elif self.slang.io_type == IOType.out:
                    self.access = (AccessType.none, AccessType.read)
                else:
                    self.access = (AccessType.read, AccessType.write)
            else:
                if self.slang.io_type == IOType.inout:
                    self.access = (AccessType.read, AccessType.none)
                elif self.slang.io_type == IOType.out:
                    self.access = (AccessType.none, AccessType.none)
                else:
                    self.access = (AccessType.read, AccessType.none)
        else:
            # todo: fwds
            self.access = (AccessType.none, AccessType.none)

    def _get_access(self, prim: PrimType) -> AccessType:
        idx: int = prim.value
        return self.access[idx]

    def gen_call_data_code(self, cg: CodeGen, context: BindContext, depth: int = 0):
        if self.children is not None:
            names: list[tuple[Any, ...]] = []
            for field, variable in self.children.items():
                variable_name = variable.gen_call_data_code(cg, context, depth+1)
                if variable_name is not None:
                    names.append(
                        (field, variable_name, variable.slang.primal_type_name, variable.slang.derivative_type_name))

            cgb = cg.call_data_structs

            cgb.begin_struct(f"_{self.variable_name}")
            for name in names:
                cgb.declare(f"_{name[1]}", name[1])

            for prim in PrimType:
                if self.access[prim.value] == AccessType.none:
                    continue

                prim_name = prim.name
                prim_type_name = self.slang.primal_type_name if prim == PrimType.primal else self.slang.derivative_type_name

                cgb.empty_line()

                cgb.empty_line()
                cgb.append_line(
                    f"void load_{prim_name}(IContext context, out {prim_type_name} value)")
                cgb.begin_block()
                for name in names:
                    cgb.declare(name[2], name[0])
                    cgb.append_statement(f"{name[1]}.load_{prim_name}(context,{name[0]})")
                    cgb.assign(f"value.{name[0]}", f"{name[0]}")
                cgb.end_block()

                cgb.empty_line()
                cgb.append_line(
                    f"void store_{prim_name}(IContext context, in {prim_type_name} value)")
                cgb.begin_block()
                for name in names:
                    cgb.append_statement(
                        f"{name[1]}.store_{prim_name}(context,value.{name[0]})")
                cgb.end_block()

            cgb.end_struct()
        else:
            if not self.transform.valid:
                return None

            # Raise error if attempting to write to non-writable type
            if self.access[0] in [AccessType.write, AccessType.readwrite] and not self.python.writable:
                if depth == 0:
                    raise ValueError(
                        f"Cannot read back value for non-writable type")

            # Generate call data
            self.python.primal.gen_calldata(cg.call_data_structs, context, self)

        if depth == 0:
            cg.call_data.declare(f"_{self.variable_name}", self.variable_name)
        return self.variable_name

    def _gen_trampoline_argument(self):
        return self.slang.gen_trampoline_argument(self.differentiable)

    def __str__(self) -> str:
        return self._recurse_str(0)

    def _recurse_str(self, depth: int) -> str:
        if self.children is not None:
            child_strs = [
                f"{'  ' * depth}{name}: {child._recurse_str(depth + 1)}" for name, child in self.children.items()]
            return "\n" + "\n".join(child_strs)
        else:
            return f"{self.python.name}"
