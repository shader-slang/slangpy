# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, Union, cast

from slangpy.core.native import Shape

import slangpy.bindings.typeregistry as tr
from slangpy.builtin.value import ValueMarshall
from slangpy.reflection import SlangType, SlangProgramLayout
from slangpy.bindings import (
    PYTHON_SIGNATURES,
    PYTHON_TYPES,
    BindContext,
    BoundVariable,
    BoundVariableRuntime,
    CodeGenBlock,
)
from slangpy import ShaderCursor, ShaderObject
from slangpy.core.native import AccessType, CallContext, NativeValueMarshall, unpack_arg
from slangpy.core.utils import is_type_castable_on_host
import slangpy.reflection as kfr


class ArrayMarshall(ValueMarshall):
    def __init__(self, layout: SlangProgramLayout, element_type: SlangType, shape: Shape):
        super().__init__(layout)
        ### Disabled as binding a struct containing an array of Tensors failed here.
        ### See https://github.com/shader-slang/slangpy/issues/255
        # if element_type.full_name == "Unknown":
        #     raise ValueError(
        #         "Element type must be fully defined. If using a Python dict, ensure it has an _type field."
        #     )

        st = element_type
        for dim in reversed(shape.as_tuple()):
            st = layout.array_type(st, dim)
        self.slang_type = st
        self.element_type = element_type
        self.concrete_shape = shape

    def reduce_type(self, context: "BindContext", dimensions: int):
        self_type = self.slang_type
        if dimensions == 0:
            return self_type
        else:
            if len(self.concrete_shape) != 1:
                raise ValueError("Vectorizing only currently supported for 1D arrays")
            if dimensions > 1:
                raise ValueError("Cannot reduce array type by more than one dimension")
            return self_type.element_type

    def resolve_type(self, context: BindContext, bound_type: "kfr.SlangType"):
        if bound_type == self.slang_type.element_type:
            return self.slang_type.element_type
        elif isinstance(bound_type, kfr.ArrayType):
            if is_type_castable_on_host(self.element_type, bound_type.element_type):
                return bound_type

        return super().resolve_type(context, bound_type)

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: "BoundVariable"):
        access = binding.access
        name = binding.variable_name
        if access[0] in [AccessType.read, AccessType.readwrite]:
            if binding.call_dimensionality == 0:
                # If not vectorizing, fallback to use of basic type as it works well
                # with Slang's implicit casts etc
                return super().gen_calldata(cgb, context, binding)
            else:
                # If vectorizing, utilize the value type.
                st = cast(kfr.ArrayType, self.slang_type)
                et = cast(SlangType, st.element_type)
                cgb.type_alias(f"_t_{name}", f"Array1DValueType<{et.full_name},{st.num_elements}>")
        else:
            cgb.type_alias(f"_t_{name}", f"NoneType")

    def build_shader_object(self, context: "BindContext", data: Any) -> "ShaderObject":
        if len(self.concrete_shape) != 1:
            return super().build_shader_object(context, data)

        unpacked = unpack_arg(data)
        st = cast(kfr.VectorType, self.slang_type)
        et = cast(SlangType, st.element_type)
        bt = context.layout.find_type_by_name(f"Array1DValueType<{et.full_name},{st.num_elements}>")
        if bt is None:
            raise ValueError(f"Could not find Slang type for {self.slang_type.full_name}")
        so = context.device.create_shader_object(bt.uniform_layout.reflection)
        cursor = ShaderCursor(so)
        cursor.write({"value": unpacked})
        return so


def _distill_array(layout: SlangProgramLayout, value: Union[list[Any], tuple[Any]]):
    from slangpy import InstanceList

    shape = (len(value),)
    while True:
        if len(value) == 0:
            return shape, tr.get_or_create_type(layout, int).slang_type

        # Get first element to decide type
        el0 = value[0]

        # If an InstanceList, return its struct type
        if isinstance(el0, InstanceList):
            et = el0._struct.struct
            return shape, et

        # Unpack from object with get_this interface
        if hasattr(value[0], "get_this"):
            el0 = value[0].get_this()

        # If a dict, check for explicit _type field
        if isinstance(el0, dict):
            tn = el0.get("_type", None)
            if tn is not None:
                et = layout.find_type_by_name(tn)
                if et is None:
                    raise ValueError(f"Could not find Slang type for '{tn}'")
                return shape, et

        # If not a list or tuple, attempt to get type from value using get_or_create_type
        if not isinstance(el0, (list, tuple)):
            et = tr.get_or_create_type(layout, type(value[0]), value[0]).slang_type
            return shape, et

        # If got here, first element is list or tuple (i.e. we have a nested array)
        N = len(value[0])
        if not all(len(x) == N for x in value):
            raise ValueError("Elements of nested array must all have equal lengths")
        shape = shape + (N,)
        value = value[0]


def python_lookup_array_type(layout: SlangProgramLayout, value: list[Any]):
    shape, et = _distill_array(layout, value)
    return ArrayMarshall(layout, cast(SlangType, et), Shape(shape))


tr.PYTHON_TYPES[list] = python_lookup_array_type
