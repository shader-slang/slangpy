from sgl import TypeReflection
from kernelfunctions.callsignature import create_slang_type_marshal
from kernelfunctions.typeregistry import SLANG_MARSHALS_BY_NAME, BaseSlangTypeMarshal


class TextureSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        if slang_type.resource_shape == TypeReflection.ResourceShape.texture_2d:
            self.container_shape = (None, None)
        else:
            raise ValueError(f"Unsupported texture shape {slang_type.resource_shape}")
        self.resource_marshal = create_slang_type_marshal(slang_type.resource_result_type)
        self.value_shape = self.resource_marshal.shape


class StructuredBufferSlangTypeMarshal(BaseSlangTypeMarshal):
    def __init__(self, slang_type: TypeReflection):
        super().__init__(slang_type)
        if slang_type.resource_shape == TypeReflection.ResourceShape.structured_buffer:
            self.container_shape = (None,)
        else:
            raise ValueError(f"Unsupported texture shape {slang_type.resource_shape}")
        self.resource_marshal = create_slang_type_marshal(slang_type.resource_result_type)
        self.value_shape = self.resource_marshal.shape


SLANG_MARSHALS_BY_NAME.update({
    "__TextureImpl": TextureSlangTypeMarshal,
    "StructuredBuffer": StructuredBufferSlangTypeMarshal,
    "RWStructuredBuffer": StructuredBufferSlangTypeMarshal,
})
