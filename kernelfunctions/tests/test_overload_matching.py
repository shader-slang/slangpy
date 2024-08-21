from typing import Any
import pytest
import kernelfunctions as kf
import sgl
import kernelfunctions.tests.helpers as helpers
import kernelfunctions.translation as kftrans


class ScalarMatchTest:
    def __init__(
        self,
        name: str,
        slang_types: list[str],
        valid_values: list[Any],
        invalid_values: list[Any],
    ) -> None:
        super().__init__()
        self.name = name
        self.slang_types = slang_types
        self.valid_values = valid_values
        self.invalid_values = invalid_values

    def __repr__(self) -> str:
        return self.name


INT_MATCHES = ScalarMatchTest(
    "int",
    ["int8_t", "int16_t", "int", "int64_t"],
    [int(1), int(-2), sgl.int1(10)],
    [float(1), sgl.float1(1.0), False, True, sgl.uint1(1)],
)
UINT_MATCHES = ScalarMatchTest(
    "uint",
    ["uint8_t", "uint16_t", "uint", "uint64_t"],
    [int(1), int(2), sgl.uint1(10)],
    [float(1), sgl.float1(1.0), False, True, sgl.int1(1)],
)
FLOAT_MATCHES = ScalarMatchTest(
    "float",
    ["half", "float", "double"],
    [float(1), sgl.float1(1.0)],
    [int(1), sgl.int1(1), False, True],
)
BOOL_MATCHES = ScalarMatchTest(
    "bool", ["bool"], [False, True, sgl.bool1(True)], [int(1), sgl.int1(1), 0.7]
)
INT2_MATCHES = ScalarMatchTest(
    "int2",
    [
        "vector<int8_t,2>",
        "vector<int16_t,2>",
        "vector<int,2>",
        "vector<int64_t,2>",
        "int2",
    ],
    [int(1), int(-2), sgl.int1(10), sgl.int2(15)],
    [float(1), sgl.float1(1.0), False, True, sgl.uint1(1), sgl.int3(1)],
)

TScalarTest = tuple[bool, str, Any, Any]
SCALAR_TESTS: list[TScalarTest] = []
for scalar_match in [
    INT_MATCHES,
    UINT_MATCHES,
    FLOAT_MATCHES,
    BOOL_MATCHES,
    INT2_MATCHES,
]:
    for slang_type_name in scalar_match.slang_types:
        scalars = scalar_match.valid_values
        for i in range(len(scalars)):
            v0 = scalars[i]
            v1 = scalars[(i + 1) % len(scalars)]
            SCALAR_TESTS.append((True, slang_type_name, v0, v1))

        scalars = scalar_match.invalid_values
        for i in range(len(scalars)):
            v0 = scalars[i]
            v1 = scalars[(i + 1) % len(scalars)]
            SCALAR_TESTS.append((False, slang_type_name, v0, v1))


def SCALAR_TEST_ID(test: str):
    return f"{test[1]}_{test[0]}_{type(test[2]).__name__}_{type(test[3]).__name__}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("test", SCALAR_TESTS, ids=SCALAR_TEST_ID)
def test_match_scalar_parameters(device_type: sgl.DeviceType, test: TScalarTest):

    device = helpers.get_device(device_type)
    succeed: bool = test[0]
    slang_type_name: str = test[1]
    v0 = test[2]
    v1 = test[3]

    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        f" void add_numbers({slang_type_name} a, {slang_type_name} b) {{ }}",
    )

    if succeed:
        match = kf.calldata.match_function_overload_to_python_args(
            function.ast_functions[0].as_function(), True, v0, v1
        )
        assert match is not None
        assert match["a"].name == "a"
        assert match["b"].name == "b"
        assert match["a"].value == v0
        assert match["b"].value == v1

        match = kf.calldata.match_function_overload_to_python_args(
            function.ast_functions[0].as_function(), True, a=v0, b=v1
        )
        assert match is not None
        assert match["a"].name == "a"
        assert match["b"].name == "b"
        assert match["a"].value == v0
        assert match["b"].value == v1

        match = kf.calldata.match_function_overload_to_python_args(
            function.ast_functions[0].as_function(), True, b=v1, a=v0
        )
        assert match is not None
        assert match["a"].name == "a"
        assert match["b"].name == "b"
        assert match["a"].value == v0
        assert match["b"].value == v1
    else:
        match = kf.calldata.match_function_overload_to_python_args(
            function.ast_functions[0].as_function(), True, v0, v1
        )
        assert match is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("test", SCALAR_TESTS, ids=SCALAR_TEST_ID)
def test_match_scalar_struct_fields(device_type: sgl.DeviceType, test: TScalarTest):

    device = helpers.get_device(device_type)
    succeed: bool = test[0]
    slang_type_name: str = test[1]
    v0 = test[2]
    v1 = test[3]

    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        f"""
struct MyStruct {{
    {slang_type_name} a;
    {slang_type_name} b;
}};
void add_numbers(MyStruct v) {{ }}
""",
    )

    if succeed:
        match = kf.calldata.match_function_overload_to_python_args(
            function.ast_functions[0].as_function(), True, {"a": v0, "b": v1}
        )
        assert match is not None
        assert isinstance(match["v"].translation_type, kftrans.StructType)
        assert match["v"].name == "v"
        assert match["v"].translation_type.fields["a"].reflection is not None
        assert match["v"].translation_type.fields["b"].reflection is not None
        assert match["v"].value["a"] == v0
        assert match["v"].value["b"] == v1
    else:
        match = kf.calldata.match_function_overload_to_python_args(
            function.ast_functions[0].as_function(), True, {"a": v0, "b": v1}
        )
        assert match is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
