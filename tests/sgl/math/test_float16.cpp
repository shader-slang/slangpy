// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/math/float16.h"

using namespace sgl;

TEST_SUITE_BEGIN("float16");

TEST_CASE("float32_to_float16")
{
    // Test cases from https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Conversions
    CHECK_EQ(math::float32_to_float16(0.0f), 0x0000);
    CHECK_EQ(math::float32_to_float16(-0.0f), 0x8000);
    CHECK_EQ(math::float32_to_float16(1.0f), 0x3c00);
    CHECK_EQ(math::float32_to_float16(-1.0f), 0xbc00);
    CHECK_EQ(math::float32_to_float16(65504.0f), 0x7bff);
}

TEST_CASE("float16_to_float32")
{
    // Test cases from https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Conversions
    CHECK_EQ(math::float16_to_float32(0x0000), 0.0f);
    CHECK_EQ(math::float16_to_float32(0x8000), -0.0f);
    CHECK_EQ(math::float16_to_float32(0x3c00), 1.0f);
    CHECK_EQ(math::float16_to_float32(0xbc00), -1.0f);
    CHECK_EQ(math::float16_to_float32(0x7bff), 65504.0f);
}

TEST_SUITE_END();
