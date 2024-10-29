

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
    def needs_specialization(self) -> bool:
        return False

    @property
    def fields(self) -> Optional[dict[str, BaseType]]:
        return None

    def update_from_bound_type(self, bound_type: 'BaseType'):
        pass

    def specialize_type(self, type: BaseType) -> Optional[BaseType]:
        return None

    def resolve_type(self, context: BindContext, bound_type: 'BaseType'):
        # default implementation of type resolution is to attempt to pass
        # either element type if it matches, or this type otherwise
        if self.element_type is not None and self.element_type.name != 'none' and len(self.element_type.get_shape()) == len(bound_type.get_shape()):
            return cast(BaseType, self.element_type)
        else:
            return self
