

from typing import Any, Optional

from sgl import Device
from kernelfunctions.shapes import TConcreteShape, TLooseOrUndefinedShape
from kernelfunctions.types import PythonMarshal, ThreadIdArg, AccessType
import kernelfunctions.codegen as cg
from kernelfunctions.types.slangmarshall import SlangMarshall


class ThreadIdArgMarshal(PythonMarshal):
    """
    Dummy class to allow for passing thread id kernel functions.
    """

    def __init__(self):
        super().__init__(ThreadIdArg)
        pass

    def get_element_shape(self, value: ThreadIdArg):
        return (value.dims,)

    def get_container_shape(self, value: ThreadIdArg):
        return ()

    def get_element_type(self, value: ThreadIdArg):
        return int

    def gen_calldata(self, slang_type_name: str, call_data_name: str, shape: TConcreteShape, access: AccessType):
        return None

    def gen_load(self, from_call_data: str, to_variable: str, to_type: SlangMarshall, transform: list[Optional[int]], access: AccessType):
        """
        Return a 1d, 2d or 3d thread id
        """
        return cg.assign(to_variable, f"{from_call_data}{self._transform_to_subscript(transform)}")

    def gen_store(self, to_call_data: str, from_variable: str, from_type: SlangMarshall, transform: list[Optional[int]], access: AccessType):
        raise NotImplementedError()

    def create_primal_calldata(self, device: Device, value: ThreadIdArg, access: AccessType):
        return None

    def read_primal_calldata(self, device: Device, call_data: Any, access: AccessType, value: ThreadIdArg):
        pass
