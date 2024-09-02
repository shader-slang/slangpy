# Simple loop to find / register all the sgl math types
import sgl
from kernelfunctions.callsignature import BasePythonTypeMarshal, register_python_type
from kernelfunctions.typemappings import is_valid_scalar_type_conversion


class SGLMathTypeMarshal(BasePythonTypeMarshal):
    def __init__(self, sgl_type: type):
        super().__init__(sgl_type)
        self.shape = sgl_type().shape
        self.element_type: type = sgl_type().element_type
        assert self.element_type in [int, float, bool]

    def is_scalar_type_compatible(self, slang_type: sgl.TypeReflection) -> bool:
        return is_valid_scalar_type_conversion(slang_type.scalar_type, self.element_type)


class SGLVectorMarshal(SGLMathTypeMarshal):
    def __init__(self, sgl_type: type):
        super().__init__(sgl_type)

    def is_compatible(self, slang_type: sgl.TypeReflection) -> bool:
        if not self.is_scalar_type_compatible(slang_type):
            return False
        if slang_type.kind == sgl.TypeReflection.Kind.vector:
            return self.shape[0] == slang_type.col_count
        elif slang_type.kind == sgl.TypeReflection.Kind.scalar:
            return self.shape[0] == 1
        return False


class SGLMatrixMarshal(SGLMathTypeMarshal):
    def __init__(self, sgl_type: type):
        super().__init__(sgl_type)

    def is_compatible(self, slang_type: sgl.TypeReflection) -> bool:
        if not self.is_scalar_type_compatible(slang_type):
            return False
        if slang_type.kind == sgl.TypeReflection.Kind.matrix:
            return self.shape[0] == slang_type.row_count and self.shape[1] == slang_type.col_count
        return False


def _reg_sgl_math_type(sgl_type: type, marshall: BasePythonTypeMarshal):
    if sgl_type is not None:
        register_python_type(
            sgl_type, marshall, lambda stream, x: stream.write(str(x) + "\n")
        )


for sgl_base_type in ["int", "float", "bool", "uint", "float16_t"]:
    for dim in range(1, 5):
        sgl_type: type = getattr(sgl.math, f"{sgl_base_type}{dim}")
        _reg_sgl_math_type(sgl_type, SGLVectorMarshal(sgl_type))
    _reg_sgl_math_type(sgl.math.quatf, SGLVectorMarshal(sgl.math.quatf))
    for row in range(2, 5):
        for col in range(2, 5):
            sgl_type: type = getattr(sgl.math, f"float{row}x{col}", None)  # type: ignore
            if sgl_type is not None:
                _reg_sgl_math_type(sgl_type, SGLMatrixMarshal(sgl_type))
