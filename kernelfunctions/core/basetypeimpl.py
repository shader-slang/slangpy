

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
        # default implementation of type resolution is to attempt to pass
        # either element type if it matches, or this type otherwise
        if self.element_type is not None and self.element_type.name == bound_type.name:
            return cast(BaseType, bound_type)
        elif bound_type.name.startswith('vector<') and self.element_type.name == bound_type.element_type.name:
            return cast(BaseType, bound_type)
        elif bound_type.name.startswith('matrix<') and self.element_type.name == bound_type.element_type.name:
            return cast(BaseType, bound_type)
        else:
            return self

    def resolve_dimensionality(self, context: BindContext, vector_target_type: 'BaseType'):
        # default implementation requires that both this type and the target type
        # have fully known element types. If so, dimensionality is just the difference
        # between the length of the 2 shapes
        if self.element_type is None:
            raise ValueError(
                f"Cannot resolve dimensionality of {self.name} without element type")
        return len(self.get_shape()) - len(vector_target_type.get_shape())
