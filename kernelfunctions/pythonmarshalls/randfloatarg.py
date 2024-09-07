

from typing import Any, Optional

from sgl import Device
import sgl
from kernelfunctions.pythonmarshalls.wanghasharg import wang_hash_code_gen
from kernelfunctions.typeregistry import register_python_type
from kernelfunctions.types import PythonMarshal, AccessType
import kernelfunctions.codegen as cg
from kernelfunctions.types.pythonmarshall import PythonDescriptor
from kernelfunctions.types.randfloatarg import RandFloatArg


class RandFloatArgMarshal(PythonMarshal):
    """
    Marshal to generate random floats
    """

    def __init__(self):
        super().__init__(RandFloatArg)
        pass

    def get_element_shape(self, value: RandFloatArg):
        return (value.dim,)

    def get_container_shape(self, value: RandFloatArg):
        return ()

    def get_element_type(self, value: RandFloatArg):
        if value.dim == 1:
            return sgl.float1
        elif value.dim == 2:
            return sgl.float2
        elif value.dim == 3:
            return sgl.float3

    def gen_calldata(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, type_name: str, variable_name: str, access: AccessType):
        cgb.add_snippet("RandFloatArg", "struct RandFloatArg {float min; float max;}")
        cgb.declare(f"RandFloatArg", variable_name)

    def gen_load(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, from_call_data: str, to_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Return a 1d, 2d or 3d hash of thread id
        """
        assert desc.element_shape is not None
        dim = desc.element_shape[0]
        assert dim is not None
        cgb.declare(f"uint{dim}", "hash_val")
        wang_hash_code_gen(cgb, dim, "hash_val")
        cgb.declarevar("randargs", from_call_data)
        cgb.declarevar("randval", f"float{dim}(hash_val%1000000)/1000000.0")
        cgb.assign(to_variable, f"randargs.min + (randargs.max - randargs.min) * randval")

    def gen_store(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, to_call_data: str, from_variable: str, transform: list[Optional[int]], access: AccessType):
        raise NotImplementedError()

    def create_primal_calldata(self, device: Device, value: RandFloatArg, access: AccessType):
        return {
            "min": value.min,
            "max": value.max,
        }

    def read_primal_calldata(self, device: Device, call_data: Any, access: AccessType, value: RandFloatArg):
        pass


register_python_type(RandFloatArg, RandFloatArgMarshal(), None)
