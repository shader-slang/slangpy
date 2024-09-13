

from typing import Any, Optional, Sequence
from kernelfunctions.types.basetype import BaseType


class BaseTypeImpl(BaseType):
    def __init__(self):
        super().__init__()

    def differentiable(self, value: Any = None):
        return False

    def differentiate(self, value: Any = None):
        return None

    def container_shape(self, value: Any = None) -> Sequence[Optional[int]]:
        return ()

    def shape(self, value: Any = None):
        return tuple(self.container_shape(value)) + tuple(self.element_type(value).shape())
