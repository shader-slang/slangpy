// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/enum.h"

TEST_SUITE_BEGIN("enum");

enum class TestEnum {
    A,
    B,
    C,
};

SGL_ENUM_INFO(
    TestEnum,
    {
        {TestEnum::A, "A"},
        {TestEnum::B, "B"},
        {TestEnum::C, "C"},
    }
);
SGL_ENUM_REGISTER(TestEnum);

enum class TestFlags {
    A = 1 << 0,
    B = 1 << 1,
    C = 1 << 2,
};
SGL_ENUM_CLASS_OPERATORS(TestFlags);

SGL_ENUM_INFO(
    TestFlags,
    {
        {TestFlags::A, "A"},
        {TestFlags::B, "B"},
        {TestFlags::C, "C"},
    }
);
SGL_ENUM_REGISTER(TestFlags);

namespace sgl {
struct TestStruct {
    enum class TestEnum {
        X,
        Y,
    };

    SGL_ENUM_INFO(
        TestEnum,
        {
            {TestEnum::X, "X"},
            {TestEnum::Y, "Y"},
        }
    );
};

SGL_ENUM_REGISTER(TestStruct::TestEnum);

static_assert(has_enum_info<void> == false);
static_assert(has_enum_info<::TestEnum> == true);
static_assert(has_enum_info<TestStruct::TestEnum> == true);

TEST_CASE("enum_has_value")
{
    CHECK(enum_has_value<TestEnum>("A"));
    CHECK(enum_has_value<TestEnum>("B"));
    CHECK(enum_has_value<TestEnum>("C"));
    CHECK_FALSE(enum_has_value<TestEnum>("D"));
}

TEST_CASE("string_to_enum")
{
    CHECK(string_to_enum<TestEnum>("A") == TestEnum::A);
    CHECK(string_to_enum<TestEnum>("B") == TestEnum::B);
    CHECK(string_to_enum<TestEnum>("C") == TestEnum::C);

    // Test enum nested in namespace and struct.
    CHECK(string_to_enum<TestStruct::TestEnum>("X") == TestStruct::TestEnum::X);
    CHECK(string_to_enum<TestStruct::TestEnum>("Y") == TestStruct::TestEnum::Y);

    // Converting unregistered strings throws.
    CHECK_THROWS(string_to_enum<TestEnum>("D"));
}

TEST_CASE("enum_to_string")
{
    CHECK(enum_to_string(TestEnum::A) == "A");
    CHECK(enum_to_string(TestEnum::B) == "B");
    CHECK(enum_to_string(TestEnum::C) == "C");

    // Test enum nested in namespace and struct.
    CHECK(enum_to_string(TestStruct::TestEnum::X) == "X");
    CHECK(enum_to_string(TestStruct::TestEnum::Y) == "Y");

    // Test converting unregistered values.
    CHECK(enum_to_string(TestEnum(-1)) == "TestEnum(-1)");
}

TEST_CASE("enum_has_value")
{
    CHECK(enum_has_value<TestStruct::TestEnum>("X"));
    CHECK(enum_has_value<TestStruct::TestEnum>("Y"));
    CHECK_FALSE(enum_has_value<TestStruct::TestEnum>("Z"));
}

TEST_CASE("flags_to_string_list")
{
    CHECK(flags_to_string_list(TestFlags{0}) == std::vector<std::string>({}));
    CHECK(flags_to_string_list(TestFlags::A) == std::vector<std::string>({"A"}));
    CHECK(flags_to_string_list(TestFlags::B) == std::vector<std::string>({"B"}));
    CHECK(flags_to_string_list(TestFlags::A | TestFlags::B) == std::vector<std::string>({"A", "B"}));
    CHECK(
        flags_to_string_list(TestFlags::A | TestFlags::B | TestFlags::C) == std::vector<std::string>({"A", "B", "C"})
    );

    // Test converting unregistered values.
    CHECK(flags_to_string_list(TestFlags::B | TestFlags(16)) == std::vector<std::string>({"B", "TestFlags(16)"}));
}

TEST_CASE("string_list_to_flags")
{
    CHECK(string_list_to_flags<TestFlags>({}) == TestFlags{0});
    CHECK(string_list_to_flags<TestFlags>({"A"}) == TestFlags::A);
    CHECK(string_list_to_flags<TestFlags>({"B"}) == TestFlags::B);
    CHECK(string_list_to_flags<TestFlags>({"A", "B"}) == (TestFlags::A | TestFlags::B));
    CHECK(string_list_to_flags<TestFlags>({"A", "B", "C"}) == (TestFlags::A | TestFlags::B | TestFlags::C));

    // Converting unregistered strings throws.
    CHECK_THROWS(string_list_to_flags<TestFlags>({"D"}));
}

} // namespace sgl

TEST_SUITE_END();
