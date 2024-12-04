# SPDX-License-Identifier: Apache-2.0
from typing import Any

from slangpy.core.native import AccessType, CallContext, Shape

import slangpy.reflection as kfr
from slangpy.backend import AccelerationStructure
from slangpy.bindings import (PYTHON_TYPES, Marshall, BindContext,
                              BoundVariable, BoundVariableRuntime,
                              CodeGenBlock)


class AccelerationStructureMarshall(Marshall):

    def __init__(self, layout: kfr.SlangProgramLayout):
        super().__init__(layout)
        st = layout.find_type_by_name("RaytracingAccelerationStructure")
        if st is None:
            raise ValueError(
                f"Could not find RaytracingAccelerationStructure slang type. This usually indicates the slangpy module has not been imported.")
        self.slang_type = st
        self.concrete_shape = Shape()

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        name = binding.variable_name
        assert isinstance(binding.vector_type, kfr.RaytracingAccelerationStructureType)
        cgb.type_alias(f"_t_{name}", f"RaytracingAccelerationStructureType")

    # Call data just returns the primal
    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: Any) -> Any:
        access = binding.access
        if access[0] != AccessType.none:
            return {
                'value': data
            }

    # Buffers just return themselves for raw dispatch
    def create_dispatchdata(self, data: Any) -> Any:
        return data


def _get_or_create_python_type(layout: kfr.SlangProgramLayout, value: AccelerationStructure):
    assert isinstance(value, AccelerationStructure)
    return AccelerationStructureMarshall(layout)


PYTHON_TYPES[AccelerationStructure] = _get_or_create_python_type
