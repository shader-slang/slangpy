from types import NoneType
from typing import Any, Optional, Union, cast
from sgl import Device, FunctionReflection, ModifierID, TypeReflection, VariableReflection

from kernelfunctions.codegen import CodeGen
from kernelfunctions.shapes import TConcreteOrUndefinedShape, TConcreteShape
from kernelfunctions.typemappings import are_element_types_compatible
from kernelfunctions.typeregistry import create_slang_type_marshal, get_python_type_marshall
from kernelfunctions.types import AccessType, IOType, CallMode


# Result of building the signature for a set of args and kwargs
# passed as part of a python call
TCallSignature = tuple[list['SignatureNode'], dict[str, 'SignatureNode']]

# Result of matching a signature to a slang function, tuple
# with set of positional arguments and optional return value
TMatchedSignature = dict[str, 'SignatureNode']


class SlangDescriptor:
    def __init__(self,
                 name: str,
                 io_type: IOType,
                 no_diff: bool,
                 primal_type: Union[TypeReflection, TypeReflection.ScalarType]) -> NoneType:
        super().__init__()
        self.name = name
        self.type = type
        self.io_type = io_type
        self.no_diff = no_diff
        self.primal_type = primal_type
        self.primal = create_slang_type_marshal(primal_type)
        self.derivative = self.primal.differentiate()

    def load_primal_fields(self):
        if isinstance(self.primal_type, TypeReflection):
            return self.primal.load_fields(self.primal_type)
        else:
            return None

    @property
    def primal_type_name(self):
        return self.primal.name

    @property
    def derivative_type_name(self):
        return self.derivative.name if self.derivative is not None else None


class SignatureNode:
    """
    Node in a built signature tree, maintains a pairing of python+slang marshall,
    and a potential set of child nodes
    """

    def __init__(self, value: Any):
        super().__init__()

        # Get the python marshall for the value + load some basic info
        self.python_marshal = get_python_type_marshall(type(value))
        self.python = self.python_marshal.get_descriptor(value)

        # Init internal data
        self.slang: SlangDescriptor = None  # type: ignore
        self.param_index = -1
        self.type_shape: Optional[list[int]] = None
        self.argument_shape: Optional[list[Optional[int]]] = None
        self.transform_inputs: TConcreteOrUndefinedShape = None
        self.transform_outputs: TConcreteOrUndefinedShape = None
        self.call_transform: Optional[list[int]] = None
        self.loadstore_transform: Optional[list[Optional[int]]] = None
        self.prim_access = AccessType.none
        self.bwds_access = (AccessType.none, AccessType.none)
        self.fwds_access = (AccessType.none, AccessType.none)
        self.variable_name = ""
        self.differentiable = False

        # Create children if value is a dict
        self.children: Optional[dict[str, SignatureNode]] = None
        if isinstance(value, dict):
            self.children = {x: SignatureNode(y) for x, y in value.items()}

    def is_compatible(
        self, slang_reflection: Union[VariableReflection, FunctionReflection, TypeReflection.ScalarType]
    ) -> bool:
        """
        Check if the node is compatible with a slang reflection
        """
        if isinstance(slang_reflection, TypeReflection.ScalarType):
            # For scalars just verifying no children atm. This happens when accessing
            # fields of vectors.
            if self.children is not None:
                return False
            return True
        else:
            # Check the element types are compatible first
            slang_type = slang_reflection.type if isinstance(
                slang_reflection, VariableReflection) else slang_reflection.return_type
            if not are_element_types_compatible(self.python.element_type, slang_type):
                return False

            # Now check children
            if self.children is not None:
                if slang_type.kind == TypeReflection.Kind.struct:
                    fields = slang_type.fields
                    if len(fields) != len(self.children):
                        return False
                    fields_by_name = {x.name: x for x in slang_type.fields}
                    for name, node in self.children.items():
                        childfield = fields_by_name.get(name, None)
                        if childfield is None:
                            return False
                        if not node.is_compatible(childfield):
                            return False
                elif slang_type.kind == TypeReflection.Kind.vector:
                    if len(self.children) != slang_type.col_count:
                        return False
                    for name, node in self.children.items():
                        if not node.is_compatible(slang_type.scalar_type):
                            return False
            return True

    def apply_signature(
        self,
        slang_reflection: Union[VariableReflection, FunctionReflection],
        path: str,
        input_transforms: Optional[dict[str, TConcreteShape]],
        output_transforms: Optional[dict[str, TConcreteShape]]
    ):
        """
        Apply a signature to the node, creating the slang marshall and calculating argument shapes
        """
        # Initial setup from properties that are only defined at top level
        if isinstance(slang_reflection, VariableReflection):
            # Function argument - check modifiers
            if slang_reflection.has_modifier(ModifierID.inout):
                io_type = IOType.inout
            elif slang_reflection.has_modifier(ModifierID.out):
                io_type = IOType.out
            else:
                io_type = IOType.inn
            no_diff = slang_reflection.has_modifier(ModifierID.nodiff)
        else:
            # Just a return value - always out, and only differentiable if function is
            io_type = IOType.out
            no_diff = not slang_reflection.has_modifier(ModifierID.differentiable)
        name = path

        # Apply the signature recursively
        self._apply_signature(
            name, io_type, no_diff,
            slang_reflection, path,
            input_transforms, output_transforms)

    def _apply_signature(
        self,
        name: str,
        io_type: IOType,
        no_diff: bool,
        slang_reflection: Union[VariableReflection, FunctionReflection, TypeReflection.ScalarType],
        path: str,
        input_transforms: Optional[dict[str, TConcreteShape]],
        output_transforms: Optional[dict[str, TConcreteShape]]
    ):
        """
        Internal function to recursively do the signature apply process
        """

        # Get slang primal type marshall
        if isinstance(slang_reflection, TypeReflection.ScalarType):
            self.slang = SlangDescriptor(name, io_type, no_diff, slang_reflection)
        elif isinstance(slang_reflection, FunctionReflection):
            self.slang = SlangDescriptor(
                name, io_type, no_diff, slang_reflection.return_type)
        else:
            self.slang = SlangDescriptor(name, io_type, no_diff, slang_reflection.type)

        # Can now decide if differentiable
        self.differentiable = not self.slang.no_diff and self.slang.derivative is not None and self.python.differentiable

        # Store some basic properties
        self.variable_name = path.replace(".", "__")

        # Calculate differentiability settings
        self._calculate_differentiability()

        # Recurse into children
        if self.children is not None:
            fields_by_name = self.slang.load_primal_fields()
            assert fields_by_name is not None
            for name, node in self.children.items():
                node.param_index = self.param_index
                node._apply_signature(
                    name, io_type, no_diff,
                    fields_by_name[name],
                    f"{path}.{name}",
                    input_transforms,
                    output_transforms)

        # If no children, this is an input, so calculate argument shape
        if self.children is None:
            if input_transforms is not None:
                self.transform_inputs = input_transforms.get(path, self.transform_inputs)
            if output_transforms is not None:
                self.transform_outputs = output_transforms.get(
                    path, self.transform_outputs)
            self._calculate_argument_shape()

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

    def write_call_data_pre_dispatch(self, device: Device, call_data: dict[str, Any], value: Any, mode: CallMode):
        """Writes value to call data dictionary pre-dispatch"""
        if self.children is not None:
            for name, child in self.children.items():
                child.write_call_data_pre_dispatch(device, call_data, value[name], mode)
        else:
            # Pick access types based on call mode.
            primal_access = self._get_primal_access(mode)
            derivative_access = self._get_derivative_access(mode)

            # Populate primal
            if primal_access != AccessType.none:
                call_data[self.variable_name + "_primal"] = self.python_marshal.create_primal_calldata(
                    device, value, primal_access)

            # Populate derivative
            if derivative_access != AccessType.none:
                call_data[self.variable_name + "_derivative"] = self.python_marshal.create_derivative_calldata(
                    device, value, derivative_access)

    def read_call_data_post_dispatch(self, device: Device, call_data: dict[str, Any], value: Any, mode: CallMode):
        """Reads value from call data dictionary post-dispatch"""
        if self.children is not None:
            for name, child in self.children.items():
                child.write_call_data_pre_dispatch(device, call_data, value[name], mode)
        else:
            # Pick access types based on call mode.
            primal_access = self._get_primal_access(mode)
            derivative_access = self._get_derivative_access(mode)

            # Populate primal
            if primal_access in [AccessType.write, AccessType.readwrite]:
                self.python_marshal.read_primal_calldata(
                    device, call_data[self.variable_name + "_primal"], primal_access, value)

            # Populate derivative
            if derivative_access in [AccessType.write, AccessType.readwrite]:
                self.python_marshal.read_derivative_calldata(
                    device, call_data[self.variable_name + "_derivative"], derivative_access, value)

    def __repr__(self):
        return self.python_marshal.__repr__()

    def _calculate_argument_shape(self):
        """
        Calculate the argument shape for the node
        - where both are defined they must match
        - where param is defined and input is not, set input to param
        - where input is defined and param is not, set param to input
        - if end up with undefined type shape, bail
        """
        input_shape = self.python.shape
        param_shape = self.slang.primal.shape
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

    def _calculate_differentiability(self):
        """
        Works out whether this node can be differentiated, then calculates the 
        corresponding access types for primitive, backwards and forwards passes
        """
        if self.differentiable:
            if self.slang.io_type == IOType.inout:
                self.prim_access = AccessType.readwrite
                self.bwds_access = (AccessType.read, AccessType.readwrite)
            elif self.slang.io_type == IOType.out:
                self.prim_access = AccessType.write
                self.bwds_access = (AccessType.none, AccessType.read)
            else:
                self.prim_access = AccessType.read
                self.bwds_access = (AccessType.read, AccessType.write)
        else:
            if self.slang.io_type == IOType.inout:
                self.prim_access = AccessType.readwrite
                self.bwds_access = (AccessType.read, AccessType.none)
            elif self.slang.io_type == IOType.out:
                self.prim_access = AccessType.write
                self.bwds_access = (AccessType.none, AccessType.none)
            else:
                self.prim_access = AccessType.read
                self.bwds_access = (AccessType.read, AccessType.none)

    def _get_primal_access(self, mode: CallMode) -> AccessType:
        if mode == CallMode.prim:
            return self.prim_access
        elif mode == CallMode.bwds:
            return self.bwds_access[0]
        else:
            return self.fwds_access[0]

    def _get_derivative_access(self, mode: CallMode) -> AccessType:
        if mode == CallMode.prim:
            return AccessType.none
        elif mode == CallMode.bwds:
            return self.bwds_access[1]
        else:
            return self.fwds_access[1]

    def allocate_output_buffer(self, device: Device):
        if self.children is not None:
            for child in self.children.values():
                child.allocate_output_buffer(device)
        else:
            if self.python_marshal.type == NoneType:
                pass

    def gen_call_data_code(self, mode: CallMode, cg: CodeGen):
        if self.children is not None:
            for child in self.children.values():
                child.gen_call_data_code(mode, cg)
        else:
            # Pick access types based on call mode.
            primal_access = self._get_primal_access(mode)
            derivative_access = self._get_derivative_access(mode)

            # Check we have a call transform (if not, this can't be an actual input node)
            assert self.loadstore_transform is not None
            cgblock = cg.call_data
            if primal_access != AccessType.none:
                assert self.slang.primal_type_name is not None
                cgblock.append_statement(
                    self.python_marshal.gen_calldata(
                        self.slang.primal_type_name, f"{self.variable_name}_primal", self.python.container_shape, primal_access)
                )
            if derivative_access != AccessType.none:
                assert self.slang.derivative_type_name is not None
                cgblock.append_statement(
                    self.python_marshal.gen_calldata(
                        self.slang.derivative_type_name, f"{self.variable_name}_derivative", self.python.container_shape, derivative_access)
                )

    def gen_load_store_code(self, mode: CallMode, cg: CodeGen):
        # Generate load store functions
        self._gen_load_primal(mode, cg)
        self._gen_load_derivative(mode, cg)
        self._gen_store_primal(mode, cg)
        self._gen_store_derivative(mode, cg)

    def _gen_load_primal(self, mode: CallMode, cg: CodeGen):
        access = self._get_primal_access(mode)
        if not access in [AccessType.read, AccessType.readwrite]:
            return None

        func_name = f"load_{self.variable_name}_primal"
        func_def = f"void {func_name}(Context context, out {self.slang.primal_type_name} val)"

        cgcode = cg.input_load_store
        if self.children is not None:
            name_to_call = {name: child._gen_load_primal(
                mode, cg) for (name, child) in self.children.items()}
            cgcode.append_line(func_def)
            cgcode.begin_block()
            for (name, child) in self.children.items():
                cgcode.append_statement(f"{name_to_call[name]}(context, val.{name})")
            cgcode.end_block()
        else:
            cgcode.append_line(func_def)
            cgcode.begin_block()
            assert self.loadstore_transform is not None
            cgcode.append_statement(
                self.python_marshal.gen_load(
                    f"call_data.{self.variable_name}_primal",
                    "val",
                    self.loadstore_transform, access))
            cgcode.end_block()

        return func_name

    def _gen_load_derivative(self, mode: CallMode, cg: CodeGen):
        access = self._get_derivative_access(mode)
        if not access in [AccessType.read, AccessType.readwrite]:
            return None

        func_name = f"load_{self.variable_name}_derivative"
        func_def = f"void {func_name}(Context context, out {self.slang.derivative_type_name} val)"

        cgcode = cg.input_load_store
        if self.children is not None:
            name_to_call = {name: child._gen_load_derivative(
                mode, cg) for (name, child) in self.children.items()}
            cgcode.append_line(func_def)
            cgcode.begin_block()
            for (name, child) in self.children.items():
                cgcode.append_statement(f"{name_to_call[name]}(context, val.{name})")
            cgcode.end_block()
        else:
            cgcode.append_line(func_def)
            cgcode.begin_block()
            assert self.loadstore_transform is not None
            cgcode.append_statement(
                self.python_marshal.gen_load(
                    f"call_data.{self.variable_name}_derivative",
                    "val",
                    self.loadstore_transform, access))
            cgcode.end_block()

        return func_name

    def _gen_store_primal(self, mode: CallMode, cg: CodeGen):
        access = self._get_primal_access(mode)
        if not access in [AccessType.write, AccessType.readwrite]:
            return None

        func_name = f"store_{self.variable_name}_primal"
        func_def = f"void {func_name}(Context context, in {self.slang.primal_type_name} val)"

        cgcode = cg.input_load_store

        if self.children is not None:
            name_to_call = {name: child._gen_store_primal(
                mode, cg) for (name, child) in self.children.items()}
            cgcode.append_line(func_def)
            cgcode.begin_block()
            for (name, child) in self.children.items():
                cgcode.append_statement(f"{name_to_call[name]}(context, val.{name})")
            cgcode.end_block()
        else:
            cgcode.append_line(func_def)
            cgcode.begin_block()
            assert self.loadstore_transform is not None
            cgcode.append_statement(
                self.python_marshal.gen_store(
                    f"call_data.{self.variable_name}_primal",
                    "val",
                    self.loadstore_transform, access))
            cgcode.end_block()
        return func_name

    def _gen_store_derivative(self, mode: CallMode, cg: CodeGen):
        access = self._get_derivative_access(mode)
        if not access in [AccessType.write, AccessType.readwrite]:
            return None

        func_name = f"store_{self.variable_name}_derivative"
        func_def = f"void {func_name}(Context context, in {self.slang.derivative_type_name} val)"

        cgcode = cg.input_load_store

        if self.children is not None:
            name_to_call = {name: child._gen_store_derivative(
                mode, cg) for (name, child) in self.children.items()}
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
            cgcode.append_statement(
                self.python_marshal.gen_store(
                    f"call_data.{self.variable_name}_derivative",
                    "val",
                    self.loadstore_transform, access))
            cgcode.end_block()

        return func_name

    def _gen_trampoline_argument(self):
        arg_def = f"{self.slang.primal.name} {self.slang.name}"
        if self.slang.io_type == IOType.inout:
            arg_def = f"inout {arg_def}"
        elif self.slang.io_type == IOType.out:
            arg_def = f"out {arg_def}"
        elif self.slang.io_type == IOType.inn:
            arg_def = f"in {arg_def}"
        if not self.differentiable:
            arg_def = f"no_diff {arg_def}"
        return arg_def
