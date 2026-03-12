// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/math/vector.h"

#include <algorithm>
#include <random>

using namespace sgl;

TEST_SUITE_BEGIN("vector");

TEST_CASE("float_formatter")
{
    float2 test0(1.23456789f, 2.f);

    CHECK_EQ(fmt::format("{}", test0), "{1.2345679, 2}");
    CHECK_EQ(fmt::format("{:e}", test0), "{1.234568e+00, 2.000000e+00}");
    CHECK_EQ(fmt::format("{:g}", test0), "{1.23457, 2}");
    CHECK_EQ(fmt::format("{:.1}", test0), "{1, 2}");
    CHECK_EQ(fmt::format("{:.3}", test0), "{1.23, 2}");
}

TEST_CASE("int_formatter")
{
    int2 test0(12, 34);

    CHECK_EQ(fmt::format("{}", test0), "{12, 34}");
    CHECK_EQ(fmt::format("{:x}", test0), "{c, 22}");
    CHECK_EQ(fmt::format("{:08x}", test0), "{0000000c, 00000022}");
    CHECK_EQ(fmt::format("{:b}", test0), "{1100, 100010}");
    CHECK_EQ(fmt::format("{:08b}", test0), "{00001100, 00100010}");
    CHECK_EQ(fmt::format("{:08X}", test0), "{0000000C, 00000022}");
}

TEST_CASE("std::less")
{
    std::vector<int2> vec{{-1, -1}, {-1, +1}, {+1, -1}, {+1, +1}, {-2, -2}, {-2, +2}, {+2, -2}, {+2, +2}};

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(vec.begin(), vec.end(), g);
    std::sort(vec.begin(), vec.end(), std::less<int2>{});
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = i + 1; j < vec.size(); ++j) {
            CHECK(std::less<int2>{}(vec[i], vec[j]));
        }
    }
}

TEST_CASE("equality_comparison")
{
    CHECK(float3(1, 2, 3) == float3(1, 2, 3));
    CHECK_FALSE(float3(1, 2, 3) == float3(1, 2, 4));
    CHECK_FALSE(float3(1, 2, 3) != float3(1, 2, 3));
    CHECK(float3(1, 2, 3) != float3(1, 3, 3));
}

TEST_CASE("lexicographic_comparison")
{
    CHECK(float3(1, 2, 3) < float3(1, 2, 4));
    CHECK(float3(1, 2, 3) < float3(2, 0, 0));
    CHECK_FALSE(float3(1, 2, 3) < float3(1, 2, 3));
    CHECK_FALSE(float3(2, 0, 0) < float3(1, 9, 9));

    CHECK(float3(1, 2, 4) > float3(1, 2, 3));
    CHECK_FALSE(float3(1, 2, 3) > float3(1, 2, 3));

    CHECK(float3(1, 2, 3) <= float3(1, 2, 3));
    CHECK(float3(1, 2, 3) <= float3(1, 2, 4));
    CHECK_FALSE(float3(1, 2, 4) <= float3(1, 2, 3));

    CHECK(float3(1, 2, 3) >= float3(1, 2, 3));
    CHECK(float3(1, 2, 4) >= float3(1, 2, 3));
    CHECK_FALSE(float3(1, 2, 3) >= float3(1, 2, 4));
}

TEST_CASE("component_wise_comparisons")
{
    using namespace math;

    CHECK(eq(float3(1, 2, 3), float3(1, 3, 3)) == bool3(true, false, true));
    CHECK(ne(float3(1, 2, 3), float3(1, 3, 3)) == bool3(false, true, false));

    CHECK(lt(float3(1, 2, 3), float3(2, 2, 2)) == bool3(true, false, false));
    CHECK(gt(float3(1, 2, 3), float3(2, 2, 2)) == bool3(false, false, true));
    CHECK(le(float3(1, 2, 3), float3(2, 2, 2)) == bool3(true, true, false));
    CHECK(ge(float3(1, 2, 3), float3(2, 2, 2)) == bool3(false, true, true));

    // vector-scalar overloads
    CHECK(eq(float3(1, 2, 3), 2.f) == bool3(false, true, false));
    CHECK(lt(float3(1, 2, 3), 2.f) == bool3(true, false, false));

    // scalar-vector overloads
    CHECK(eq(2.f, float3(1, 2, 3)) == bool3(false, true, false));
    CHECK(gt(2.f, float3(1, 2, 3)) == bool3(true, false, false));
}

TEST_SUITE_END();
