

from typing import Union
from sgl import TypeReflection

from kernelfunctions.typeregistry import SLANG_MARSHALS_BY_KIND, SLANG_MARSHALS_BY_SCALAR_TYPE, AccessType, BaseSlangTypeMarshal, create_slang_type_marshal


class ScalarSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: Union[TypeReflection, TypeReflection.ScalarType]):
        super().__init__(slang_type)
        self.value_shape = (1,)

    @property
    def differentiable(self) -> bool:
        return self.scalar_type in [TypeReflection.ScalarType.float16, TypeReflection.ScalarType.float32, TypeReflection.ScalarType.float64]

    def differentiate(self):
        return self if self.differentiable else None

    def to_slang(self, access: AccessType):
        return self.name


class VectorSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (slang_type.col_count,)
        self.dtype = create_slang_type_marshal(slang_type.scalar_type)

    @property
    def differentiable(self):
        return self.dtype.differentiable

    def differentiate(self):
        return self if self.differentiable else None

    def load_fields(self, slang_type: TypeReflection):
        return dict(zip(["x", "y", "z", "w"][:slang_type.col_count], [slang_type.scalar_type] * slang_type.col_count))


class MatrixSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (slang_type.row_count, slang_type.col_count)
        self.dtype = create_slang_type_marshal(slang_type.scalar_type)

    @property
    def differentiable(self):
        return self.dtype.differentiable

    def differentiate(self):
        return self if self.differentiable else None


class StructSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        self.value_shape = (1,)

    def load_fields(self, slang_type: TypeReflection):
        return {x.name: x for x in slang_type.fields}


SLANG_MARSHALS_BY_KIND.update({
    TypeReflection.Kind.scalar: ScalarSlangTypeMarshal,
    TypeReflection.Kind.vector: VectorSlangTypeMarshal,
    TypeReflection.Kind.matrix: MatrixSlangTypeMarshal,
    TypeReflection.Kind.struct: StructSlangTypeMarshal,
})
for x in TypeReflection.ScalarType:
    SLANG_MARSHALS_BY_SCALAR_TYPE[x] = ScalarSlangTypeMarshal
