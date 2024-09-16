from typing import Optional, TYPE_CHECKING

from kernelfunctions.core.enums import PrimType

if TYPE_CHECKING:
    from kernelfunctions.core.basetype import BaseType


class BaseVariable:
    def __init__(self):
        super().__init__()
        self.param_index = -1
        self.name = ""
        self.primal: 'BaseType' = None  # type: ignore (init in constructor)
        self.derivative: Optional['BaseType'] = None
        self.fields: Optional[dict[str, 'BaseVariable']] = None

    def __repr__(self) -> str:
        return self._recurse_str(0)

    def is_compatible(self, other: 'BaseVariable') -> bool:
        raise NotImplementedError()

    @property
    def primal_type_name(self):
        raise NotImplementedError()

    @property
    def derivative_type_name(self):
        raise NotImplementedError()

    @property
    def primal_element_name(self):
        raise NotImplementedError()

    @property
    def derivative_element_name(self):
        raise NotImplementedError()

    @property
    def root_element_name(self):
        raise NotImplementedError()

    @property
    def argument_declaration(self):
        raise NotImplementedError()

    def get(self, t: PrimType):
        raise NotImplementedError()

    def _recurse_str(self, depth: int) -> str:
        raise NotImplementedError()
