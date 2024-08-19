from typing import Optional, Type

N = object()

TValueType = Type


class BaseValue:
    def __init__(
        self,
        name: str,
        value_type: TValueType,
        dimensionality: Optional[int] = None,
        shape: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        self.name = name
        self.value_type = value_type
        if dimensionality is not None and shape is not None:
            raise ValueError("Cannot specify both dimensionality and shape")
        if dimensionality is None and shape is None:
            raise ValueError("Must specify either dimensionality or shape")
        if dimensionality is not None:
            self.shape = (N,) * dimensionality
        else:
            self.shape = shape


# Fixed size scaler value such as a float, 1D with fixed length of 1
class ScalarValue(BaseValue):
    def __init__(self, name: str, value_type: TValueType):
        super().__init__(name, value_type, shape=(1,))


# Fixed size vector value such as a float3, 1D with fixed length of num components
class VectorValue(BaseValue):
    def __init__(self, name: str, value_type: TValueType, num_components: int):
        super().__init__(name, value_type, shape=(num_components,))


# Fixed size matrix value such as a float3x3, 2D with fixed length of num rows and num columns
class MatrixValue(BaseValue):
    def __init__(
        self, name: str, value_type: TValueType, num_rows: int, num_columns: int
    ):
        super().__init__(name, value_type, shape=(num_rows, num_columns))


# Buffer value such as a StructuredBuffer<float>, 1D with undefined length
class BufferValue(BaseValue):
    def __init__(self, name: str, value_type: TValueType):
        super().__init__(name, value_type, dimensionality=1)


# Texture value such as a Texture2D<float>, 2D with undefined length
class TextureValue(BaseValue):
    def __init__(self, name: str, value_type: TValueType):
        super().__init__(name, value_type, dimensionality=2)


# Tensor value such as a Tensor<float>, ND with undefined length
class TensorValue(BaseValue):
    def __init__(self, name: str, value_type: TValueType, dimensionality: int):
        super().__init__(name, value_type, dimensionality=dimensionality)
