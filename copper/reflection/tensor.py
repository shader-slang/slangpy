from __future__ import annotations

from .base import SlangType, SlangName, opaque_type

import enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..layers import ReflectedType


class TensorKind(enum.Enum):
    Tensor = enum.auto()
    RWTensor = enum.auto()
    DiffTensor = enum.auto()
    DiffRWTensor = enum.auto()
    GradTensor = enum.auto()

    def __str__(self):
        if self is TensorKind.Tensor:
            return "Tensor"
        if self is TensorKind.RWTensor:
            return "RWTensor"
        if self is TensorKind.DiffTensor:
            return "DiffTensor"
        if self is TensorKind.DiffRWTensor:
            return "DiffRWTensor"
        if self is TensorKind.GradTensor:
            return "GradTensor"
        raise RuntimeError("Invalid tensor kind")

    def writeable(self) -> bool:
        return (
            self is TensorKind.RWTensor
            or self is TensorKind.DiffRWTensor
            or self is TensorKind.GradTensor
        )

    def differentiable(self) -> bool:
        return self is TensorKind.DiffTensor or self is TensorKind.DiffRWTensor


@opaque_type("Tensor", "RWTensor", "DiffTensor")
@opaque_type("Tensor1D", "Tensor2D", "Tensor3D", "Tensor4D")
@opaque_type("RWTensor1D", "RWTensor2D", "RWTensor3D", "RWTensor4D")
@opaque_type("DiffTensor1D", "DiffTensor2D", "DiffTensor3D", "DiffTensor4D")
# @opaque_type('DiffRWTensor', 'DiffRWTensor1D', 'DiffRWTensor2D', 'DiffRWTensor3D', 'DiffRWTensor4D')
# @opaque_type('GradTensor')
class TensorType(SlangType):
    def __init__(self, kind: TensorKind, dtype: SlangType, ndim: int):
        super().__init__()
        self.kind = kind
        self.dtype = dtype
        self.ndim = ndim

    def __str__(self) -> str:
        out = str(self.kind)
        if self.ndim <= 4:
            out += f"{self.ndim}D"
        out += "<" + str(self.dtype)
        if self.ndim > 4:
            out += f", {self.ndim}"
        out += ">"
        return out

    @staticmethod
    def from_reflection(name: SlangName, type: ReflectedType) -> TensorType:
        args = type.generic_args()
        assert len(args) >= 1
        basename = name.base

        if any(basename.endswith(nd) for nd in ("1D", "2D", "3D", "4D")):
            ndim = int(basename[-2])
            basename = basename[:-2]
        else:
            assert len(args) >= 2 and isinstance(args[1], int)
            ndim = args[1]

        dtype = args[0]
        assert isinstance(dtype, SlangType)

        result = TensorType(TensorKind[basename], dtype, ndim)

        return result
