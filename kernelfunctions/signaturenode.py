from enum import Enum
from typing import Any, Optional, Union, cast
from sgl import FunctionReflection, ModifierID, TypeReflection, VariableReflection

from kernelfunctions.codegen import CodeGen, declare
from kernelfunctions.shapes import TConcreteOrUndefinedShape, TConcreteShape
from kernelfunctions.typemappings import are_element_types_compatible
from kernelfunctions.typeregistry import AccessType, BaseSlangTypeMarshal, create_slang_type_marshal, get_python_type_marshall


class CallMode(Enum):
    prim = 0
    bwds = 1
    fwds = 2


class IOType(Enum):
    none = 0
    inn = 1
    out = 2
    inout = 3


# Result of building the signature for a set of args and kwargs
# passed as part of a python call
TCallSignature = tuple[list['SignatureNode'], dict[str, 'SignatureNode']]

# Result of matching a signature to a slang function, tuple
# with set of positional arguments and optional return value
TMatchedSignature = dict[str, 'SignatureNode']


class SignatureNode:
    """
    Node in a built signature tree, maintains a pairing of python+slang marshall,
    and a potential set of child nodes
    """

    def __init__(self, value: Any):
        super().__init__()

        # Get the python marshall for the value + load some basic info
        self.python_marshal = get_python_type_marshall(value)
        self.python_container_shape = self.python_marshal.get_container_shape(value)
        self.python_element_shape = self.python_marshal.get_element_shape(value)
        self.python_element_type = self.python_marshal.get_element_type(value)
        self.python_differentiable = self.python_marshal.is_differentiable(value)

        # Calculate combined element and container shape
        python_shape = ()
        if self.python_container_shape is not None:
            python_shape += self.python_container_shape
        if self.python_element_shape is not None:
            python_shape += self.python_element_shape
        self.python_shape = python_shape if len(python_shape) > 0 else None

        # Init internal data
        self.slang_primal: BaseSlangTypeMarshal = None  # type: ignore
        self.param_index = -1
        self.type_shape: Optional[list[int]] = None
        self.argument_shape: Optional[list[Optional[int]]] = None
        self.transform_inputs: TConcreteOrUndefinedShape = None
        self.transform_outputs: TConcreteOrUndefinedShape = None
        self.call_transform: Optional[list[int]] = None
        self.io_type = IOType.none
        self.no_diff = False
        self.slang_differential = None
        self.prim_access = AccessType.none
        self.bwds_access = (AccessType.none, AccessType.none)
        self.fwds_access = (AccessType.none, AccessType.none)
        self.name = ""
        self.path = ""

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
            if not are_element_types_compatible(self.python_element_type, slang_type):
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
                self.io_type = IOType.inout
            elif slang_reflection.has_modifier(ModifierID.out):
                self.io_type = IOType.out
            else:
                self.io_type = IOType.inn
            self.no_diff = slang_reflection.has_modifier(ModifierID.nodiff)
        else:
            # Just a return value - always out, and only differentiable if function is
            self.io_type = IOType.out
            self.no_diff = not slang_reflection.has_modifier(ModifierID.differentiable)
        self.name = path

        # Apply the signature recursively
        self._apply_signature(slang_reflection, path, input_transforms, output_transforms)

    def _apply_signature(
        self,
        slang_reflection: Union[VariableReflection, FunctionReflection, TypeReflection.ScalarType],
        path: str,
        input_transforms: Optional[dict[str, TConcreteShape]],
        output_transforms: Optional[dict[str, TConcreteShape]]
    ):
        """
        Internal function to recursively do the signature apply process
        """

        # Store path
        self.path = path

        # Get slang primal type marshall
        if isinstance(slang_reflection, TypeReflection.ScalarType):
            self.slang_primal = create_slang_type_marshal(slang_reflection)
        else:
            slang_type = slang_reflection.type if isinstance(
                slang_reflection, VariableReflection) else slang_reflection.return_type
            self.slang_primal = create_slang_type_marshal(slang_type)

        # Get slang differential type marshall from the primal
        if not self.no_diff and self.python_differentiable:
            self.slang_differential = self.slang_primal.differentiate()

        # Calculate differentiability settings
        self._calculate_differentiability()

        # Recurse into children
        if self.children is not None:
            fields_by_name = self.slang_primal.load_fields(slang_type)
            for name, node in self.children.items():
                node.param_index = self.param_index
                node.io_type = self.io_type
                node.no_diff = self.no_diff
                node.name = name
                node._apply_signature(
                    fields_by_name[name], f"{path}.{name}", input_transforms, output_transforms)

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

    def write_call_data_pre_dispatch(self, call_data: dict[str, Any], value: Any):
        """Writes value to call data dictionary pre-dispatch"""
        pass

    def read_call_data_post_dispatch(self, call_data: dict[str, Any], value: Any):
        """Reads value from call data dictionary post-dispatch"""
        pass

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
        assert self.slang_primal is not None
        input_shape = self.python_shape
        param_shape = self.slang_primal.shape
        if input_shape is not None:
            # Optionally use the input remap to re-order input dimensions
            if self.transform_inputs is not None:
                if not self.python_container_shape:
                    raise ValueError(
                        f"Input transforms can only be applied to container types")
                if len(self.transform_inputs) != len(self.python_container_shape):
                    raise ValueError(
                        f"Input remap {self.transform_inputs} is different to the container shape {self.python_container_shape}"
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
        if self.slang_differential is not None:
            if self.io_type == IOType.inout:
                self.prim_access = AccessType.readwrite
                self.bwds_access = (AccessType.read, AccessType.readwrite)
            elif self.io_type == IOType.out:
                self.prim_access = AccessType.write
                self.bwds_access = (AccessType.none, AccessType.read)
            else:
                self.prim_access = AccessType.read
                self.bwds_access = (AccessType.read, AccessType.write)
        else:
            if self.io_type == IOType.inout:
                self.prim_access = AccessType.readwrite
                self.bwds_access = (AccessType.read, AccessType.none)
            elif self.io_type == IOType.out:
                self.prim_access = AccessType.write
                self.bwds_access = (AccessType.none, AccessType.none)
            else:
                self.prim_access = AccessType.read
                self.bwds_access = (AccessType.read, AccessType.none)

    def gen_code_for_input(self, mode: CallMode, cg: CodeGen):

        # Check we have a call transform (if not, this can't be an actual input node)
        assert self.call_transform is not None

        # Build some basic properties from node info.
        name = self.path.replace(".", "__")
        primal_type = self.slang_primal.name
        derivative_type = None if self.slang_differential is None else self.slang_differential.name
        container_shape = self.python_container_shape

        # Pick access types based on call mode.
        if mode == CallMode.prim:
            primal_access = self.prim_access
            derivative_access = AccessType.none
        elif mode == CallMode.bwds:
            primal_access, derivative_access = self.bwds_access
        else:
            primal_access, derivative_access = self.fwds_access

        # Get the indexers for accessing call data
        if container_shape is not None:
            if self.transform_inputs is not None:
                transform = [self.call_transform[self.transform_inputs[i]]
                             for i in range(len(container_shape))]
            else:
                transform = [self.call_transform[i] for i in range(len(container_shape))]
        else:
            transform = [0]

        primal_index = self.python_marshal.get_indexer(transform, primal_access)
        derivative_index = self.python_marshal.get_indexer(transform, derivative_access)

        # Generate members of the call data structure
        cgblock = cg.call_data
        if primal_access != AccessType.none:
            assert primal_type is not None
            cgblock.append_statement(declare(self.python_marshal.get_calldata_typename(
                primal_type, container_shape, primal_access), f"{name}_primal"))
        if derivative_access != AccessType.none:
            assert derivative_type is not None
            cgblock.append_statement(declare(self.python_marshal.get_calldata_typename(
                derivative_type, container_shape, derivative_access), f"{name}_derivative"))

        # Generate primal load function
        cgblock = cg.input_load_store
        if primal_access == AccessType.read or primal_access == AccessType.readwrite:
            cgblock.append_line(
                f"void load_{name}_primal(int[] call_id, out {primal_type} val)")
            cgblock.begin_block()
            cgblock.append_statement(f"val = call_data.{name}_primal{primal_index}")
            cgblock.end_block()

        # Generate derivative load function
        cgblock = cg.input_load_store
        if derivative_access == AccessType.read or derivative_access == AccessType.readwrite:
            cgblock.append_line(
                f"void load_{name}_derivative(int[] call_id, out {derivative_type} val)")
            cgblock.begin_block()
            cgblock.append_statement(
                f"val = call_data.{name}_derivative{derivative_index}")
            cgblock.end_block()

        # Generate primal store function
        cgblock = cg.input_load_store
        if primal_access == AccessType.write or primal_access == AccessType.readwrite:
            cgblock.append_line(
                f"void store_{name}_primal(int[] call_id, in {primal_type} val)")
            cgblock.begin_block()
            cgblock.append_statement(f"call_data.{name}_primal{primal_index} = val")
            cgblock.end_block()

        # Generate derivative store function
        cgblock = cg.input_load_store
        if derivative_access == AccessType.write or derivative_access == AccessType.readwrite:
            cgblock.append_line(
                f"void store_{name}_derivative(int[] call_id, in {derivative_type} val)")
            cgblock.begin_block()
            cgblock.append_statement(
                f"call_data.{name}_derivative{derivative_index} = val")
            cgblock.end_block()

    def typename_primal(self):
        """
        Get the typename for the primal value
        """
        return self.slang_primal.name

    def typename_derivative(self):
        """
        Get the typename for the derivative value
        """
        return self.slang_primal.name + ".Differential"

    def valuename_primal(self):
        """
        Get the value name for the primal value
        """
        return self.name

    def valuename_derivative(self):
        """
        Get the value name for the derivative value
        """
        return self.name + "_grad"

    def declare_primal(self):
        """
        Declare the node in slang
        """
        return declare(self.typename_primal(), self.valuename_primal())

    def declare_derivative(self):
        """
        Declare the node in slang
        """
        return declare(self.typename_derivative(), self.valuename_derivative())
