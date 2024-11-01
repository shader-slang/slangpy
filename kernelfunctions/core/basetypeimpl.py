

from typing import Optional, cast

from .basetype import BaseType, BindContext


class BaseTypeImpl(BaseType):
    def __init__(self):
        super().__init__()

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

    def resolve_type(self, context: BindContext, bound_type: 'BaseType'):

        # if implicit element casts enabled, allow conversion from type to element type
        if context.options['implicit_element_casts']:
            if self.element_type is not None and self.element_type.name == bound_type.name:
                return cast(BaseType, bound_type)

        # TODO: move to tensor type
        # if implicit tensor casts enabled, allow conversion from vector/matrix to element type
        if context.options['implicit_tensor_casts']:
            if bound_type.name.startswith('vector<') and self.element_type.name == bound_type.element_type.name:
                return cast(BaseType, bound_type)
            elif bound_type.name.startswith('matrix<') and self.element_type.name == bound_type.element_type.name:
                return cast(BaseType, bound_type)

        # Default to just casting to itself (i.e. no implicit cast)
        return self

    def resolve_dimensionality(self, context: BindContext, vector_target_type: 'BaseType'):
        # default implementation requires that both this type and the target type
        # have fully known element types. If so, dimensionality is just the difference
        # between the length of the 2 shapes
        if self.element_type is None:
            raise ValueError(
                f"Cannot resolve dimensionality of {self.name} without element type")
        return len(self.get_shape()) - len(vector_target_type.get_shape())
