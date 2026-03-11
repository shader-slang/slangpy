# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, Optional, cast

from slangpy.core.native import Shape, NativeMarshall

import slangpy.bindings.typeregistry as tr
from slangpy.bindings import PYTHON_TYPES, BindContext, BoundVariable, can_direct_bind_common
from slangpy.reflection import SlangProgramLayout, SlangType, UnknownType, StructType, InterfaceType
from slangpy.core.native import AccessType

from .value import ValueMarshall
import slangpy.reflection.vectorize as spyvec


class StructMarshall(ValueMarshall):

    def __init__(
        self,
        layout: SlangProgramLayout,
        fields: dict[str, NativeMarshall],
        slang_type: Optional[SlangType] = None,
    ):
        super().__init__(layout)
        if slang_type is None:
            slang_type = layout.find_type_by_name("Unknown")
            if slang_type is None:
                raise ValueError(
                    f"Could not find Struct slang type. This usually indicates the slangpy module has not been imported."
                )
        self.slang_type = slang_type
        self.concrete_shape = Shape()
        self._fields = fields

    @property
    def has_derivative(self) -> bool:
        return True

    @property
    def is_writable(self) -> bool:
        return True

    def resolve_types(self, context: BindContext, bound_type: "SlangType"):
        # Support this struct being of unknown type, which like a scalar, just means
        # we're attempting to bind the value as is. This is especially important
        # for structs, as they may be SOA types with fields that need to be
        # bound individually.
        if (
            isinstance(self.slang_type, UnknownType)
            and not isinstance(bound_type, (UnknownType, InterfaceType))
            and not bound_type.is_generic
        ):
            return [bound_type]

        # Support resolving generic struct
        as_struct = spyvec.struct_to_struct(self.slang_type, bound_type)
        if as_struct is not None:
            return [as_struct]

        # Support resolving generic vector (occurs if user attempts to provide a vector
        # by specifying a dictionary with x,y,z... fields)
        as_vector = spyvec.vector_to_vector(self.slang_type, bound_type)
        if as_vector is not None:
            return [as_vector]

        return None

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: SlangType,
    ):
        if not binding.children or not self._fields:
            return 0
        return max(
            cast(int, binding.children[name].call_dimensionality) for name in self._fields.keys()
        )

    def can_direct_bind(self, binding: "BoundVariable") -> bool:
        if binding.children is not None:
            return (
                binding.call_dimensionality == 0
                and not binding.create_param_block
                and binding.vector_type is not None
                and binding.access[0] == AccessType.read
                and all(child.direct_bind for child in binding.children.values())
            )
        return can_direct_bind_common(binding)

    # A struct type should get a dictionary, and just return that for raw dispatch

    def gen_trampoline_load(
        self, cgb: "CodeGenBlock", binding: "BoundVariable", data_name: str, value_name: str
    ) -> bool:
        if not binding.direct_bind:
            return False
        return super().gen_trampoline_load(cgb, binding, data_name, value_name)

    def gen_trampoline_store(
        self, cgb: "CodeGenBlock", binding: "BoundVariable", data_name: str, value_name: str
    ) -> bool:
        if not binding.direct_bind:
            return False
        return super().gen_trampoline_store(cgb, binding, data_name, value_name)

    def create_dispatchdata(self, data: Any) -> Any:
        if isinstance(data, dict):
            return data
        else:
            raise ValueError(f"Expected dictionary for struct type, got {type(data)}")


def create_vr_type_for_value(layout: SlangProgramLayout, value: dict[str, Any]):
    assert isinstance(value, dict)
    slang_type: Optional[SlangType] = None

    if "_type" in value:
        type_name = value["_type"]
        slang_type = layout.find_type_by_name(type_name)
        if slang_type is None:
            raise ValueError(f"Could not find type {type_name}")

    fields = {
        name: tr.get_or_create_type(layout, type(val), val)
        for name, val in value.items()
        if name != "_type"
    }

    return StructMarshall(layout, fields, slang_type)


PYTHON_TYPES[dict] = create_vr_type_for_value
