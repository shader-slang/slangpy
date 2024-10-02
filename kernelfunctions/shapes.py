from typing import Optional, Sequence, TypedDict, cast

TConcreteShape = tuple[int, ...]
TConcreteOrUndefinedShape = Optional[TConcreteShape]
TLooseShape = tuple[int, ...]
TLooseOrUndefinedShape = Optional[TLooseShape]

TArgShapesResult = TypedDict(
    "TArgShapesResult",
    {
        "type_shapes": list[list[int]],
        "arg_shapes": list[list[int]],
        "call_shape": list[int],
    },
)


def check_concrete(shape: Optional[Sequence[Optional[int]]]) -> TConcreteShape:
    assert shape is not None
    assert not None in shape
    assert not -1 in shape
    return cast(TConcreteShape, shape)
