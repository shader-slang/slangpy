from typing import Any
import pytest
from kernelfunctions.backend import DeviceType, float1, int1, int2, bool1, uint1, int3
from kernelfunctions.callsignature import CallMode, build_signature, build_signature_hash, match_signature, apply_signature
import kernelfunctions.tests.helpers as helpers


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
    [int(1), int(-2), int1(10)],
    [float(1), float1(1.0), False, True, uint1(1)],
)
UINT_MATCHES = ScalarMatchTest(
    "uint",
    ["uint8_t", "uint16_t", "uint", "uint64_t"],
    [int(1), int(2), uint1(10)],
    [float(1), float1(1.0), False, True, int1(1)],
)
FLOAT_MATCHES = ScalarMatchTest(
    "float",
    ["half", "float", "double"],
    [float(1), float1(1.0)],
    [int(1), int1(1), False, True],
)
BOOL_MATCHES = ScalarMatchTest(
    "bool", ["bool"], [False, True, bool1(True)], [int(1), int1(1), 0.7]
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
    [int2(15)],
    [int(1), int(-2), int1(10), float(1), float1(1.0),
     False, True, uint1(1), int3(1)],
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


def calc_vector_name(slang_type_name: str) -> str:
    import re
    match = re.match(r"^(float|int|uint|bool)([1-4]+)$", slang_type_name)
    if match:
        return f"vector<{match.group(1)},{match.group(2)}>"
    return slang_type_name


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("test", SCALAR_TESTS, ids=SCALAR_TEST_ID)
def test_match_scalar_parameters(device_type: DeviceType, test: TScalarTest):

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

    cs_hash_0 = build_signature_hash(v0, v1)
    cs_hash_1 = build_signature_hash(a=v0, b=v1)
    cs_hash_2 = build_signature_hash(b=v1, a=v0)
    assert cs_hash_0 != cs_hash_1
    assert cs_hash_0 != cs_hash_2
    assert cs_hash_1 == cs_hash_2

    if succeed:

        sig = build_signature(v0, v1)
        match = match_signature(
            sig, function.overloads[0], CallMode.prim)
        assert match is not None
        assert match["a"].python_marshal.type == type(v0)
        assert match["b"].python_marshal.type == type(v1)
        apply_signature(match, function.ast_functions[0].as_function(), CallMode.prim)
        assert match["a"].slang.primal
        assert match["b"].slang.primal
        assert match["a"].slang.primal.name == calc_vector_name(slang_type_name)
        assert match["b"].slang.primal.name == calc_vector_name(slang_type_name)

        sig = build_signature(a=v0, b=v1)
        match = match_signature(
            sig, function.ast_functions[0].as_function(), CallMode.prim)
        assert match is not None
        assert match["a"].python_marshal.type == type(v0)
        assert match["b"].python_marshal.type == type(v1)
        apply_signature(match, function.ast_functions[0].as_function(), CallMode.prim)
        assert match["a"].slang.primal
        assert match["b"].slang.primal
        assert match["a"].slang.primal.name == calc_vector_name(slang_type_name)
        assert match["b"].slang.primal.name == calc_vector_name(slang_type_name)

        sig = build_signature(b=v1, a=v0)
        match = match_signature(
            sig, function.ast_functions[0].as_function(), CallMode.prim)
        assert match is not None
        assert match["a"].python_marshal.type == type(v0)
        assert match["b"].python_marshal.type == type(v1)
        apply_signature(match, function.ast_functions[0].as_function(), CallMode.prim)
        assert match["a"].slang.primal
        assert match["b"].slang.primal
        assert match["a"].slang.primal.name == calc_vector_name(slang_type_name)
        assert match["b"].slang.primal.name == calc_vector_name(slang_type_name)

    else:
        sig = build_signature(v0, v1)
        match = match_signature(
            sig, function.ast_functions[0].as_function(), CallMode.prim)
        assert match is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("test", SCALAR_TESTS, ids=SCALAR_TEST_ID)
def test_match_scalar_struct_fields(device_type: DeviceType, test: TScalarTest):

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
        sig = build_signature({"a": v0, "b": v1})
        match = match_signature(
            sig, function.ast_functions[0].as_function(), CallMode.prim)
        assert match is not None
        assert match["v"].python_marshal.type == dict
        assert match["v"].children
        assert match["v"].children["a"].python_marshal.type == type(v0)
        assert match["v"].children["b"].python_marshal.type == type(v1)

        apply_signature(match, function.ast_functions[0].as_function(), CallMode.prim)
        assert match["v"].children
        assert match["v"].slang.primal
        assert match["v"].slang.primal.name == "MyStruct"
        assert match["v"].children["a"].slang.primal
        assert match["v"].children["b"].slang.primal
        assert match["v"].children["a"].slang.primal.name == calc_vector_name(
            slang_type_name)
        assert match["v"].children["b"].slang.primal.name == calc_vector_name(
            slang_type_name)

    else:
        sig = build_signature({"a": v0, "b": v1})
        match = match_signature(
            sig, function.ast_functions[0].as_function(), CallMode.prim)
        assert match is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
