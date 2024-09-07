# Simple loop to find / register all the sgl math types
from typing import Any
import sgl
from kernelfunctions.typemappings import TSGLVector
from kernelfunctions.typeregistry import register_python_type
from kernelfunctions.types import PythonMarshal
import numpy as np
import numpy.typing as npt

ETYPE_TO_NP = {
    int: np.int32,
    float: np.float32,
    bool: np.int32,
}


class SGLMathTypeMarshal(PythonMarshal):
    def __init__(self, sgl_type: type):
        super().__init__(sgl_type)
        self.shape = sgl_type().shape

    def get_element_shape(self, value: Any) -> tuple[int | None, ...]:
        return self.shape


class SGLVectorMarshal(SGLMathTypeMarshal):
    def __init__(self, sgl_type: type):
        super().__init__(sgl_type)
        self.npt = ETYPE_TO_NP[sgl_type().element_type]  # type: ignore

    def primal_to_numpy(self, value: TSGLVector):
        return np.array([value[i] for i in range(value.shape[0])], dtype=self.npt)

    def primal_from_numpy(self, value: npt.NDArray[Any]) -> TSGLVector:
        return self.type(list(value.view(self.npt)))


class SGLMatrixMarshal(SGLMathTypeMarshal):
    def __init__(self, sgl_type: type):
        super().__init__(sgl_type)

    def primal_to_numpy(self, value: Any):
        return value.to_numpy()

    def primal_from_numpy(self, value: npt.NDArray[Any]) -> Any:
        return self.type(value)


def _reg_sgl_math_type(sgl_type: type, marshall: PythonMarshal):
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
