from types import NoneType
from typing import Any, Optional, cast

from kernelfunctions.backend import Device
from kernelfunctions.codegen import CodeGen
from kernelfunctions.shapes import TConcreteOrUndefinedShape, TConcreteShape
from kernelfunctions.types import AccessType, IOType, CallMode
from kernelfunctions.types.basevalue import BaseValue
from kernelfunctions.types.enums import PrimType
from kernelfunctions.types.pythonvalue import PythonValue
from kernelfunctions.types.slangvalue import SlangValue

# Result of matching a signature to a slang function, tuple
# with set of positional arguments and optional return value
TMatchedSignature = dict[str, BaseValue]

TMatchedNodes = dict[str, 'SignatureNode']


class SignatureCall:
    def __init__(self) -> NoneType:
        super().__init__()
        self.args: list['SignatureNode'] = []
        self.kwargs: dict[str, 'SignatureNode'] = {}

    def values(self) -> list['SignatureNode']:
        return self.args + list(self.kwargs.values())


class SignatureNode:
    """
    Node in a built signature tree, maintains a pairing of python+slang marshall,
    and a potential set of child nodes
    """

    def __init__(self, python: PythonValue, slang: SlangValue,
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
        self.differentiable = not self.slang.no_diff and self.slang.derivative is not None and self.python.differentiable

        # Store some basic properties
        self.variable_name = self.path.replace(".", "__")

        # Calculate differentiability settings
        self._calculate_differentiability(mode)

        # Create children if python value has children
        self.children: Optional[dict[str, SignatureNode]] = None
        if python.fields is not None:
            assert slang.fields is not None
            self.children = {}
            for name, child_python in python.fields.items():
                child_slang = slang.fields[name]
                self.children[name] = SignatureNode(
                    child_python, child_slang, mode, input_transforms, output_transforms, self.path)

        # If no children, this is an input, so calculate argument shape
        if self.children is None:
            if input_transforms is not None:
                self.transform_inputs = input_transforms.get(
                    self.path, self.transform_inputs)
            if output_transforms is not None:
                self.transform_outputs = output_transforms.get(
                    self.path, self.transform_outputs)
            self._calculate_argument_shape()

#
#    def is_compatible(
#        self, slang_value: SlangValue
#    ) -> bool:
#        """
#        Check if the node is compatible with a slang value
#        """
#        if isinstance(slang_reflection, TypeReflection.ScalarType):
#            # For scalars just verifying no children atm. This happens when accessing
#            # fields of vectors.
#            if self.children is not None:
#                return False
#            return True
#        else:
#            # Check the element types are compatible first
#            slang_type = slang_reflection.type if isinstance(
#                slang_reflection, VariableReflection) else slang_reflection.return_type
#            if not are_element_types_compatible(self.python.element_type, slang_type):
#                return False
#
#            # Now check children
#            if self.children is not None:
#                if slang_type.kind == TypeReflection.Kind.struct:
#                    fields = slang_type.fields
#                    if len(fields) != len(self.children):
#                        return False
#                    fields_by_name = {x.name: x for x in slang_type.fields}
#                    for name, node in self.children.items():
#                        childfield = fields_by_name.get(name, None)
#                        if childfield is None:
#                            return False
#                        if not node.is_compatible(childfield):
#                            return False
#                elif slang_type.kind == TypeReflection.Kind.vector:
#                    if len(self.children) != slang_type.col_count:
#                        return False
#                    for name, node in self.children.items():
#                        if not node.is_compatible(slang_type.scalar_type):
#                            return False
#            return True
#

    def get_input_list(self, args: list['SignatureNode']):
        """
        Recursively populate flat list of argument nodes
        """
        self._get_input_list_recurse(args)
        return args

    def _get_input_list_recurse(self, args: list['SignatureNode']):
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
            for name, child in self.children.items():
                child.write_call_data_pre_dispatch(device, call_data, value[name])
        else:
            call_data = self.python.create_calldata(device, self.access, value)

    def read_call_data_post_dispatch(self, device: Device, call_data: dict[str, Any], value: Any):
        """Reads value from call data dictionary post-dispatch"""
        if self.children is not None:
            for name, child in self.children.items():
                child.read_call_data_post_dispatch(device, call_data, value[name])
        else:
            self.python.read_calldata(device, self.access, value, call_data)

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

    def gen_call_data_code(self, cg: CodeGen):
        if self.children is not None:
            for child in self.children.values():
                child.gen_call_data_code(cg)
        else:
            assert self.loadstore_transform is not None
            self.python.gen_calldata(
                cg.call_data,
                self.variable_name,
                self.access)

    def gen_load_store_code(self, cg: CodeGen):
        # Generate load store functions
        self._gen_load(cg, PrimType.primal)
        self._gen_load(cg, PrimType.derivative)
        self._gen_store(cg, PrimType.primal)
        self._gen_store(cg, PrimType.derivative)

    def _gen_load(self, cg: CodeGen, prim: PrimType):
        access = self._get_access(prim)
        if not access in [AccessType.read, AccessType.readwrite]:
            return None

        prim_name = prim.name
        func_name = f"load_{self.variable_name}_{prim_name}"
        func_def = f"void {func_name}(Context context, out {self.slang.primal_type_name} val)"

        cgcode = cg.input_load_store
        if self.children is not None:
            name_to_call = {name: child._gen_load(
                cg, prim) for (name, child) in self.children.items()}
            cgcode.append_line(func_def)
            cgcode.begin_block()
            for (name, child) in self.children.items():
                cgcode.append_statement(f"{name_to_call[name]}(context, val.{name})")
            cgcode.end_block()
        else:
            cgcode.append_line(func_def)
            cgcode.begin_block()
            assert self.loadstore_transform is not None
            self.python.gen_load(
                cgcode,
                f"call_data.{self.variable_name}_{prim_name}",
                "val",
                self.loadstore_transform,
                prim,
                access)
            cgcode.end_block()

        return func_name

    def _gen_store(self, cg: CodeGen, prim: PrimType):
        access = self._get_access(prim)
        if not access in [AccessType.write, AccessType.readwrite]:
            return None

        prim_name = prim.name
        func_name = f"store_{self.variable_name}_{prim_name}"
        func_def = f"void {func_name}(Context context, in {self.slang.get(prim).name} val)"

        cgcode = cg.input_load_store

        if self.children is not None:
            name_to_call = {name: child._gen_store(
                cg, prim) for (name, child) in self.children.items()}
            cgcode.append_line(func_def)
            cgcode.begin_block()
            for (name, child) in self.children.items():
                n = name_to_call[name]
                if n is not None:
                    cgcode.append_statement(f"{n}(context, val.{name})")
                else:
                    cgcode.append_line(f"// {name} not writable")
            cgcode.end_block()
        else:
            cgcode.append_line(func_def)
            cgcode.begin_block()
            assert self.loadstore_transform is not None
            self.python.gen_store(
                cgcode,
                "val",
                f"call_data.{self.variable_name}_{prim_name}",
                self.loadstore_transform,
                prim,
                access)
            cgcode.end_block()
        return func_name

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
