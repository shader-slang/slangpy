// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/maths.h"

#include <cstdint>

using namespace sgl;

TEST_SUITE_BEGIN("maths");

TEST_CASE("is_power_of_two")
{
    CHECK(is_power_of_two(0));

    for (uint32_t i = 1; i != 0; i <<= 1) {
        CAPTURE(i);
        CHECK(is_power_of_two(i));
    }

    for (uint32_t i = 4; i != 0; i <<= 1) {
        CAPTURE(i);
        CHECK(!is_power_of_two(i + 1));
        CHECK(!is_power_of_two(i - 1));
    }
}

TEST_CASE("div_round_up")
{
    CHECK_EQ(div_round_up(0, 1), 0);
    CHECK_EQ(div_round_up(1, 1), 1);
    CHECK_EQ(div_round_up(2, 1), 2);

    CHECK_EQ(div_round_up(0, 4), 0);
    CHECK_EQ(div_round_up(1, 4), 1);
    CHECK_EQ(div_round_up(3, 4), 1);
    CHECK_EQ(div_round_up(4, 4), 1);
    CHECK_EQ(div_round_up(5, 4), 2);

    CHECK_EQ(div_round_up(0, 7), 0);
    CHECK_EQ(div_round_up(1, 7), 1);
    CHECK_EQ(div_round_up(6, 7), 1);
    CHECK_EQ(div_round_up(7, 7), 1);
    CHECK_EQ(div_round_up(8, 7), 2);

    CHECK_EQ(div_round_up(0, 8), 0);
    CHECK_EQ(div_round_up(1, 8), 1);
    CHECK_EQ(div_round_up(7, 8), 1);
    CHECK_EQ(div_round_up(8, 8), 1);
    CHECK_EQ(div_round_up(9, 8), 2);
}

TEST_CASE("align_to")
{
    CHECK_EQ(align_to(1, 0), 0);
    CHECK_EQ(align_to(1, 1), 1);
    CHECK_EQ(align_to(1, 2), 2);

    CHECK_EQ(align_to(4, 0), 0);
    CHECK_EQ(align_to(4, 1), 4);
    CHECK_EQ(align_to(4, 3), 4);
    CHECK_EQ(align_to(4, 4), 4);
    CHECK_EQ(align_to(4, 5), 8);

    CHECK_EQ(align_to(7, 0), 0);
    CHECK_EQ(align_to(7, 1), 7);
    CHECK_EQ(align_to(7, 6), 7);
    CHECK_EQ(align_to(7, 7), 7);
    CHECK_EQ(align_to(7, 8), 14);

    CHECK_EQ(align_to(8, 0), 0);
    CHECK_EQ(align_to(8, 1), 8);
    CHECK_EQ(align_to(8, 7), 8);
    CHECK_EQ(align_to(8, 8), 8);
    CHECK_EQ(align_to(8, 9), 16);
}

TEST_SUITE_END();
