

from .basetype import BaseType, BindContext
from .reflection import SlangType, SlangProgramLayout


class BaseTypeImpl(BaseType):
    def __init__(self, layout: 'SlangProgramLayout'):
        super().__init__(layout)

    @property
    def has_derivative(self) -> bool:
        return False

    @property
    def is_writable(self) -> bool:
        return False

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        # Default to just casting to itself (i.e. no implicit cast)
        return self.slang_type

    def resolve_dimensionality(self, context: BindContext, vector_target_type: 'SlangType'):
        # Default implementation requires that both this type and the target type
        # have fully known element types. If so, dimensionality is just the difference
        # between the length of the 2 shapes
        if self.slang_type is None:
            raise ValueError(
                f"Cannot resolve dimensionality of {type(self)} without slang type")
        return len(self.slang_type.shape) - len(vector_target_type.shape)
