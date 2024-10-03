

from .basetype import BaseType


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

    def update_from_bound_type(self, bound_type: 'BaseType'):
        pass
