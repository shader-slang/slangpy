# Simple loop to find / register all the sgl math types
import sgl
from kernelfunctions.callsignature import BasePythonTypeMarshal, register_python_type


class SGLVectorMarshal(BasePythonTypeMarshal):
    def __init__(self, sgl_type: type):
        super().__init__(sgl_type)


class SGLMatrixMarshal(BasePythonTypeMarshal):
    def __init__(self, sgl_type: type):
        super().__init__(sgl_type)


def _reg_sgl_math_type(sgl_type: type, marshall: BasePythonTypeMarshal):
    if sgl_type is not None:
        register_python_type(
            sgl_type, marshall, lambda stream, x: stream.write(str(x) + "\n")
        )


for sgl_base_type in ["int", "float", "bool", "uint", "float16_t"]:
    for dim in range(1, 5):
        sgl_type: type = getattr(sgl_base_type, f"{sgl_base_type}{dim}")
        _reg_sgl_math_type(sgl_type, SGLVectorMarshal(sgl_type))
    _reg_sgl_math_type(sgl.math.quatf, SGLVectorMarshal(sgl.math.quatf))
    for row in range(2, 5):
        for col in range(2, 5):
            sgl_type: type = getattr(sgl.math, f"float{row}x{col}")
            _reg_sgl_math_type(sgl_type, SGLMatrixMarshal(sgl_type))
