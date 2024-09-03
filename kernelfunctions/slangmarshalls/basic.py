

from sgl import TypeReflection
from kernelfunctions.callsignature import SLANG_MARSHALS_BY_KIND, BaseSlangTypeMarshal


class ScalarSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (1,)


class VectorSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (slang_type.col_count,)


class MatrixSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (slang_type.row_count, slang_type.col_count)


class StructSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (1,)


SLANG_MARSHALS_BY_KIND.update({
    TypeReflection.Kind.scalar: ScalarSlangTypeMarshal,
    TypeReflection.Kind.vector: VectorSlangTypeMarshal,
    TypeReflection.Kind.matrix: MatrixSlangTypeMarshal,
    TypeReflection.Kind.struct: StructSlangTypeMarshal,
})
