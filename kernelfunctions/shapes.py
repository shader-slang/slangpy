from typing import TYPE_CHECKING, TypedDict, Union

if TYPE_CHECKING:
    from kernelfunctions.core import NativeShape

TArgShapesResult = TypedDict(
    "TArgShapesResult",
    {
        "type_shapes": list[list[int]],
        "arg_shapes": list[list[int]],
        "call_shape": list[int],
    },
)

TShapeOrTuple = Union[tuple[int, ...], 'NativeShape']


def check_concrete(shape: 'NativeShape') -> 'NativeShape':
    assert shape.shape is not None
    assert not None in shape.shape
    assert not -1 in shape.shape
    return shape
