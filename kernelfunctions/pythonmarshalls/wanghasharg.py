

from typing import Any, Optional

import kernelfunctions.codegen as cg

from kernelfunctions.backend import Device, uint1, uint2, uint3
from kernelfunctions.typeregistry import register_python_type
from kernelfunctions.types import PythonMarshal, AccessType
from kernelfunctions.types.enums import PrimType
from kernelfunctions.types.pythonmarshall import PythonDescriptor
from kernelfunctions.types.wanghasharg import WangHashArg

WANG_RANDOM_ARG = r"""
uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}
"""


def wang_hash_code_gen(cgb: cg.CodeGenBlock, dim: int, to_variable: str):
    cgb.add_snippet("wang_hash", WANG_RANDOM_ARG)  # add wang hash snippet
    if dim == 1:
        cgb.assign(to_variable, f"wang_hash(context.thread_id.x)")
    elif dim == 2:
        cgb.declarevar("x", "wang_hash(context.thread_id.x)")
        cgb.declarevar("y", "wang_hash(x)")
        cgb.assign(to_variable, f"uint2(x,y)")
    elif dim == 3:
        cgb.declarevar("x", "wang_hash(context.thread_id.x)")
        cgb.declarevar("y", "wang_hash(x)")
        cgb.declarevar("z", "wang_hash(y)")
        cgb.assign(to_variable, f"uint3(x,y,z)")


class WangHashArgMarshal(PythonMarshal):
    """
    Marshall to do code gen for generating wang hashes and feeding into
    unsigned integer vectors.
    """

    def __init__(self):
        super().__init__(WangHashArg)
        pass

    def get_element_shape(self, value: WangHashArg):
        return (value.dims,)

    def get_container_shape(self, value: WangHashArg):
        return ()

    def get_element_type(self, value: WangHashArg):
        if value.dims == 1:
            return uint1
        elif value.dims == 2:
            return uint2
        elif value.dims == 3:
            return uint3

    def gen_calldata(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, type_name: str, variable_name: str, access: AccessType):
        pass

    def gen_load(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, from_call_data: str, to_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Return a 1d, 2d or 3d hash of thread id
        """
        assert desc.element_shape is not None
        dim = desc.element_shape[0]
        assert dim is not None
        wang_hash_code_gen(cgb, dim, to_variable)

    def gen_store(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, to_call_data: str, from_variable: str, transform: list[Optional[int]], access: AccessType):
        raise NotImplementedError()

    def create_calldata(self, device: Device, value: WangHashArg, access: AccessType, prim: PrimType):
        assert prim == PrimType.primal
        return None

    def read_calldata(self, device: Device, call_data: Any, access: AccessType, prim: PrimType, value: WangHashArg):
        assert prim == PrimType.primal
        pass


register_python_type(WangHashArg, WangHashArgMarshal(), None)
