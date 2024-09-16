from .base import SlangType

from typing import Any


class ResourceType(SlangType):
    pass


class TextureType(ResourceType):
    def __init__(self, dimensions: int, writeable: bool):
        super().__init__()
        self.dimensions = dimensions
        self.writeable = writeable

    def __str__(self) -> str:
        raise NotImplementedError

    def __eq__(self, other: Any):
        if not isinstance(other, TextureType):
            return NotImplemented
        return self.dimensions == other.dimensions and self.writeable == other.writeable

    def __repr__(self):
        return f"TextureType(dimensions={self.dimensions})"


class StructuredBufferType(ResourceType):
    def __init__(self, dtype: SlangType, writeable: bool):
        super().__init__()
        self.dtype = dtype
        self.writeable = writeable

    def __str__(self) -> str:
        return f'{"RW" if self.writeable else ""}StructuredBuffer<{self.dtype}>'

    def __eq__(self, other: Any):
        if not isinstance(other, StructuredBufferType):
            return NotImplemented
        return self.writeable == other.writeable and self.dtype == other.dtype

    def __repr__(self):
        return f"StructuredBufferType(dtype={repr(self.dtype)}, writeable={self.writeable})"


class RawBufferType(ResourceType):
    def __init__(self, writeable: bool):
        super().__init__()
        self.writeable = writeable

    def __str__(self) -> str:
        return f'{"RW" if self.writeable else ""}ByteAddressBuffer'

    def __eq__(self, other: Any):
        if not isinstance(other, RawBufferType):
            return NotImplemented
        return self.writeable == other.writeable

    def __repr__(self):
        return f"RawBufferType(writeable={self.writeable})"
