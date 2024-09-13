

from typing import Any, Optional, Sequence
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.basevalue import BaseVariable
from kernelfunctions.types.enums import AccessType, PrimType


class BaseTypeImpl(BaseType):
    def __init__(self):
        super().__init__()

    def differentiable(self, value: Any = None):
        return False

    def differentiate(self, value: Any = None):
        return None

    def container_shape(self, value: Any = None) -> Sequence[Optional[int]]:
        return ()

    def shape(self, value: Any = None):
        return tuple(self.container_shape(value)) + tuple(self.element_type(value).shape())

    # Load should only ever be reading the primal directly from the call data
    def gen_load_store(self, cgb: CodeGenBlock, input_value: 'BaseVariable', name: str, transform: list[Optional[int]],  access: tuple[AccessType, AccessType]):
        cgb.begin_struct(f"_{name}")
        cgb.type_alias(f"primal_type", input_value.primal_element_name)
        cgb.type_alias(f"derivative_type", input_value.derivative_element_name)
        for prim in PrimType:
            prim_name = prim.name
            prim_access = access[prim.value]
            if prim_access in [AccessType.read, AccessType.readwrite]:
                cgb.append_line(
                    f"static void load_{prim_name}(Context context, out {prim_name}_type value) {{ call_data.{name}.load_{prim_name}(context,value); }}")
            if prim_access in [AccessType.write, AccessType.readwrite]:
                cgb.append_line(
                    f"static void store_{prim_name}(Context context, in {prim_name}_type value) {{ call_data.{name}.store_{prim_name}(context,value); }}")
        cgb.end_struct()
