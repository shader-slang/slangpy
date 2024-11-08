from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .basetype import BaseType


class BaseVariable:
    def __init__(self):
        super().__init__()
        self.primal: 'BaseType'
        self.fields: Optional[dict[str, 'BaseVariable']] = None

    def __repr__(self) -> str:
        return self._recurse_str(0)

    def _recurse_str(self, depth: int) -> str:
        raise NotImplementedError()
