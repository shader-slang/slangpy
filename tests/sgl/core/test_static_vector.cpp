// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/static_vector.h"

using namespace sgl;

TEST_SUITE_BEGIN("static_vector");

TEST_CASE("default_constructor")
{
    static_vector<int, 4> v;
    CHECK(v.empty());
    CHECK_EQ(v.size(), 0);
    CHECK_EQ(v.capacity(), 4);
    CHECK_EQ(v.begin(), v.end());
}

TEST_CASE("size_constructor")
{
    static_vector<int, 4> v(3, 0);
    CHECK_EQ(v.size(), 3);
    CHECK_EQ(v.capacity(), 4);
    CHECK_EQ(v[0], 0);
    CHECK_EQ(v[1], 0);
    CHECK_EQ(v[2], 0);
}

TEST_CASE("initializer_list_constructor")
{
    static_vector<int, 8> v1 = {1, 2, 3, 4};
    CHECK_EQ(v1.size(), 4);
    CHECK_EQ(v1.capacity(), 8);
    CHECK_EQ(v1[0], 1);
    CHECK_EQ(v1[1], 2);
    CHECK_EQ(v1[2], 3);
    CHECK_EQ(v1[3], 4);
}

TEST_CASE("element_access")
{
    static_vector<int, 4> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);
    CHECK_EQ(v[0], 1);
    CHECK_EQ(v[1], 2);
    CHECK_EQ(v[2], 3);
    CHECK_EQ(v[3], 4);
    CHECK_EQ(v.front(), 1);
    CHECK_EQ(v.back(), 4);
    CHECK_EQ(*v.begin(), 1);
    CHECK_EQ(*(v.end() - 1), 4);
    CHECK_EQ(v.data()[0], 1);
    CHECK_EQ(v.data()[1], 2);
    CHECK_EQ(v.data()[2], 3);
    CHECK_EQ(v.data()[3], 4);
}

TEST_CASE("push_back")
{
    static_vector<int, 4> v;
    v.push_back(1);
    CHECK_EQ(v.size(), 1);
    CHECK_EQ(v.capacity(), 4);
    CHECK_EQ(v[0], 1);
    v.push_back(2);
    CHECK_EQ(v.size(), 2);
    CHECK_EQ(v.capacity(), 4);
    CHECK_EQ(v[0], 1);
    CHECK_EQ(v[1], 2);
    v.push_back(3);
    CHECK_EQ(v.size(), 3);
    CHECK_EQ(v.capacity(), 4);
    CHECK_EQ(v[0], 1);
    CHECK_EQ(v[1], 2);
    CHECK_EQ(v[2], 3);
    v.push_back(4);
    CHECK_EQ(v.size(), 4);
    CHECK_EQ(v.capacity(), 4);
    CHECK_EQ(v[0], 1);
    CHECK_EQ(v[1], 2);
    CHECK_EQ(v[2], 3);
    CHECK_EQ(v[3], 4);
}

TEST_CASE("clear")
{
    static_vector<int, 4> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);
    CHECK_EQ(v.size(), 4);
    CHECK_EQ(v.capacity(), 4);
    v.clear();
    CHECK_EQ(v.size(), 0);
    CHECK_EQ(v.capacity(), 4);
}

TEST_SUITE_END();
