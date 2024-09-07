
from typing import Union


class WangRandomArg:
    """
    Dummy class to allow for passing random numbers in using wang hashing
    """

    def __init__(self, dims: int = 3, element_type: type[Union[int, float]] = int):
        super().__init__()
        self.dims = dims
        self.element_type = element_type
