

from typing import Any, Optional

import kernelfunctions.codegen as cg

from kernelfunctions.backend import Device, int1, int2, int3
from kernelfunctions.typeregistry import register_python_type
from kernelfunctions.types import PythonMarshal, ThreadIdArg, AccessType
from kernelfunctions.types.enums import PrimType
from kernelfunctions.types.pythonmarshall import PythonDescriptor


class ThreadIdArgMarshal(PythonMarshal):
    """
    Marshal to copy thread id directly into variable
    """

    def __init__(self):
        super().__init__(ThreadIdArg)
        pass

    def get_element_shape(self, value: ThreadIdArg):
        return (value.dims,)

    def get_container_shape(self, value: ThreadIdArg):
        return ()

    def get_element_type(self, value: ThreadIdArg):
        if value.dims == 1:
            return int1
        elif value.dims == 2:
            return int2
        elif value.dims == 3:
            return int3

    def gen_calldata(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, type_name: str, variable_name: str, access: AccessType):
        pass

    def gen_load(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, from_call_data: str, to_variable: str, transform: list[Optional[int]], access: AccessType):
        """
        Return a 1d, 2d or 3d thread id
        """
        assert desc.element_shape is not None
        dim = desc.element_shape[0]
        if dim == 1:
            cgb.assign(to_variable, f"context.thread_id.x")
        elif dim == 2:
            cgb.assign(to_variable, f"context.thread_id.xy")
        elif dim == 3:
            cgb.assign(to_variable, f"context.thread_id.xyz")

    def gen_store(self, cgb: cg.CodeGenBlock, desc: PythonDescriptor, to_call_data: str, from_variable: str, transform: list[Optional[int]], access: AccessType):
        raise NotImplementedError()

    def create_calldata(self, device: Device, value: ThreadIdArg, access: AccessType, prim: PrimType):
        assert prim == PrimType.primal
        return None

    def read_calldata(self, device: Device, call_data: Any, access: AccessType, prim: PrimType, value: ThreadIdArg):
        assert prim == PrimType.primal
        pass


register_python_type(ThreadIdArg, ThreadIdArgMarshal(), None)
