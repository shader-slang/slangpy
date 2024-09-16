from types import NoneType
from typing import Any, Optional, cast

from kernelfunctions.backend import Device
from kernelfunctions.codegen import CodeGen
from kernelfunctions.shapes import TConcreteOrUndefinedShape, TConcreteShape
from kernelfunctions.types import AccessType, IOType, CallMode
from kernelfunctions.types.enums import PrimType
from kernelfunctions.core.pythonvariable import PythonVariable
from kernelfunctions.core.slangvariable import SlangVariable


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
                 input_transforms: Optional[dict[str, TConcreteShape]] = None,
                 output_transforms: Optional[dict[str, TConcreteShape]] = None,
                 path: Optional[str] = None):

        super().__init__()

        # Store the python and slang marshall
        self.python = python
        self.slang = slang

        # Initialize path
        if path is None:
            self.path = self.slang.name
            self.python.name = self.slang.name
        else:
            self.path = f"{path}.{self.python.name}"

        # Get the python marshall for the value + load some basic info
        self.param_index = -1
        self.type_shape: Optional[list[int]] = None
        self.argument_shape: Optional[list[Optional[int]]] = None
        self.transform_inputs: TConcreteOrUndefinedShape = None
        self.transform_outputs: TConcreteOrUndefinedShape = None
        self.call_transform: Optional[list[int]] = None
        self.loadstore_transform: Optional[list[Optional[int]]] = None
        self.access = (AccessType.none, AccessType.none)
        self.variable_name = ""

        # Can now decide if differentiable
        self.differentiable = not self.slang.no_diff and self.slang.derivative is not None and self.python.differentiable and self.python.has_derivative

        # Store some basic properties
        self.variable_name = self.path.replace(".", "__")

        # Calculate differentiability settings
        self._calculate_differentiability(mode)

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

        # If no children, this is an input, so calculate argument shape
        if self.children is None:
            if input_transforms is not None:
                self.transform_inputs = input_transforms.get(
                    self.path, self.transform_inputs)
            if output_transforms is not None:
                self.transform_outputs = output_transforms.get(
                    self.path, self.transform_outputs)
            self._calculate_argument_shape()

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
        if self.type_shape is not None:
            args.append(self)

    def write_call_data_pre_dispatch(self, device: Device, call_data: dict[str, Any], value: Any):
        """Writes value to call data dictionary pre-dispatch"""
        if self.children is not None:
            res = {}
            for name, child in self.children.items():
                child.write_call_data_pre_dispatch(device, res, value[name])
            if len(res) > 0:
                call_data[self.variable_name] = res
        else:
            cd_val = self.python.create_calldata(
                device, self.access, value)
            if cd_val is not None:
                call_data[self.variable_name] = cd_val

    def read_call_data_post_dispatch(self, device: Device, call_data: dict[str, Any], value: Any):
        """Reads value from call data dictionary post-dispatch"""
        if self.children is not None:
            for name, child in self.children.items():
                child_cd = call_data.get(child.variable_name, None)
                if child_cd is not None:
                    child.read_call_data_post_dispatch(device, child_cd, value[name])
        else:
            cd_val = call_data.get(self.variable_name, None)
            if cd_val is not None:
                self.python.read_calldata(device, self.access, value, cd_val)

    def __repr__(self):
        return self.python.__repr__()

    def _calculate_argument_shape(self):
        """
        Calculate the argument shape for the node
        - where both are defined they must match
        - where param is defined and input is not, set input to param
        - where input is defined and param is not, set param to input
        - if end up with undefined type shape, bail
        """
        input_shape = self.python.shape
        param_shape = self.slang.primal.shape()
        if input_shape is not None:
            # Optionally use the input remap to re-order input dimensions
            if self.transform_inputs is not None:
                if not self.python.container_shape:
                    raise ValueError(
                        f"Input transforms can only be applied to container types")
                if len(self.transform_inputs) != len(self.python.container_shape):
                    raise ValueError(
                        f"Input remap {self.transform_inputs} is different to the container shape {self.python.container_shape}"
                    )
                new_input_shape = list(input_shape)
                for i in self.transform_inputs:
                    new_input_shape[i] = input_shape[self.transform_inputs[i]]
                input_shape = new_input_shape

            # Now assign out shapes, accounting for differing dimensionalities
            type_len = len(param_shape)
            input_len = len(input_shape)
            type_end = type_len - 1
            input_end = input_len - 1
            new_param_type_shape: list[int] = []
            for i in range(type_len):
                param_dim_idx = type_end - i
                input_dim_idx = input_end - i
                param_dim_size = param_shape[param_dim_idx]
                input_dim_size = input_shape[input_dim_idx]
                if param_dim_size is not None and input_dim_size is not None:
                    if param_dim_size != input_dim_size:
                        raise ValueError(
                            f"Arg {self.param_index}, PS[{param_dim_idx}] != IS[{input_dim_idx}], {param_dim_size} != {input_dim_size}"
                        )
                    new_param_type_shape.append(param_dim_size)
                elif param_dim_size is not None:
                    new_param_type_shape.append(param_dim_size)
                elif input_dim_size is not None:
                    new_param_type_shape.append(input_dim_size)
                else:
                    raise ValueError(f"Arg {self.param_index} type shape is ambiguous")
            new_param_type_shape.reverse()
            self.type_shape = new_param_type_shape
            self.argument_shape = list(input_shape[: input_len - type_len])
        else:
            # If input not defined, parameter shape is the argument shape
            if None in param_shape:
                raise ValueError(f"Arg {self.param_index} type shape is ambiguous")
            self.type_shape = list(cast(TConcreteShape, param_shape))
            self.argument_shape = None

        if self.argument_shape is None:
            return

        # Verify transforms match argument shape
        if self.transform_outputs is not None and len(self.transform_outputs) != len(self.argument_shape):
            raise ValueError(
                f"Transform outputs {self.transform_outputs} must have the same number of dimensions as the argument shape {self.argument_shape}")

        # Define a default function transform which basically maps argument
        # dimensions to call dimensions 1-1, with a bit of extra work to handle
        # arguments that aren't the same size or shapes that aren't defined.
        # This is effectively what numpy does.
        self.call_transform = [i for i in range(len(self.argument_shape))]

        # Inject any custom transforms
        if self.transform_outputs is not None:
            for i in range(len(self.argument_shape)):
                if self.transform_outputs[i] is not None:
                    self.call_transform[i] = self.transform_outputs[i]

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

    def gen_call_data_code(self, cg: CodeGen, depth: int = 0):
        if self.children is not None:
            names: list[tuple[str, str]] = []
            for field, variable in self.children.items():
                variable_name = variable.gen_call_data_code(cg, depth+1)
                if variable_name is not None:
                    names.append((field, variable_name))

            cgb = cg.call_data_structs

            if self.access[1] == AccessType.none:
                cgb.begin_struct(
                    f"_{self.variable_name}: IValueCallData<{self.slang.primal_type_name}>")
            else:
                cgb.begin_struct(
                    f"_{self.variable_name}: ICallData<{self.slang.primal_type_name},{self.slang.derivative_type_name}>")

            for name in names:
                cgb.declare(f"_{name[1]}", name[1])

            for prim in PrimType:
                if self.access[prim.value] == AccessType.none:
                    continue

                prim_name = prim.name
                prim_type_name = self.slang.primal_type_name if prim == PrimType.primal else self.slang.derivative_type_name

                cgb.empty_line()
                cgb.type_alias(f"{prim_name}_type", prim_type_name)

                cgb.empty_line()
                cgb.append_line(
                    f"void load_{prim_name}(IContext context, out {prim_name}_type value)")
                cgb.begin_block()
                for name in names:
                    cgb.declare(f"_{name[1]}::{prim_name}_type", f"{name[0]}")
                    cgb.append_statement(f"{name[1]}.load_{prim_name}(context,{name[0]})")
                    cgb.assign(f"value.{name[0]}", f"{name[0]}")
                cgb.end_block()

                cgb.empty_line()
                cgb.append_line(
                    f"void store_{prim_name}(IContext context, in {prim_name}_type value)")
                cgb.begin_block()
                for name in names:
                    cgb.append_statement(
                        f"{name[1]}.store_{prim_name}(context,value.{name[0]})")
                cgb.end_block()

            cgb.end_struct()
        else:
            if self.loadstore_transform is None:
                return None

            # Raise error if attempting to write to non-writable type
            if self.access[0] in [AccessType.write, AccessType.readwrite] and not self.python.writable:
                raise ValueError(
                    f"Cannot read back value for non-writable type")

            # Generate call data
            self.python.gen_calldata(
                cg.call_data_structs,
                self.variable_name,
                self.loadstore_transform,
                self.access)
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
