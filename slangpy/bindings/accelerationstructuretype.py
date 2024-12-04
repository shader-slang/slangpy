

from typing import Any, Optional

from slangpy.backend import ResourceUsage, Buffer
from slangpy.core import BindContext, BaseType, BaseTypeImpl, BoundVariable, CodeGenBlock, AccessType, BoundVariableRuntime, CallContext, Shape
from slangpy.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES
import slangpy.core.reflection as kfr
from sgl import AccelerationStructure


class AccelerationStructureType(BaseTypeImpl):

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
    return AccelerationStructureType(layout)


PYTHON_TYPES[AccelerationStructure] = _get_or_create_python_type