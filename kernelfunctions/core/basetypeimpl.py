

from typing import Optional, cast

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

    @property
    def differentiable(self):
        return False

    @property
    def derivative(self):
        return None

    @property
    def fields(self) -> Optional[dict[str, BaseType]]:
        return None

    def get_slang_type(self, context: 'BindContext') -> 'SlangType':
        t = self.slang_type
        assert t
        return t

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):

        # if implicit element casts enabled, allow conversion from type to element type
        if context.options['implicit_element_casts']:
            if self.slang_type.element_type == bound_type:
                return bound_type

        # TODO: move to tensor type
        # if implicit tensor casts enabled, allow conversion from vector/matrix to element type
        if context.options['implicit_tensor_casts']:
            if bound_type.full_name.startswith('vector<') and self.slang_type.element_type == bound_type.element_type:
                return bound_type
            elif bound_type.full_name.startswith('matrix<') and self.slang_type.element_type == bound_type.element_type:
                return bound_type

        # Default to just casting to itself (i.e. no implicit cast)
        return self.slang_type

    def resolve_dimensionality(self, context: BindContext, vector_target_type: 'SlangType'):
        # default implementation requires that both this type and the target type
        # have fully known element types. If so, dimensionality is just the difference
        # between the length of the 2 shapes
        if self.slang_type is None:
            raise ValueError(
                f"Cannot resolve dimensionality of {self.name} without slang type")
        return len(self.get_shape(None)) - len(vector_target_type.shape)
