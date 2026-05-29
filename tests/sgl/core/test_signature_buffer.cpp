// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/signature_buffer.h"

#include <string>
#include <type_traits>

using namespace sgl;

static_assert(!std::is_copy_constructible_v<SignatureBuffer>);
static_assert(!std::is_copy_assignable_v<SignatureBuffer>);

TEST_SUITE_BEGIN("signature_buffer");

TEST_CASE("append_values")
{
    SignatureBuffer buffer;

    buffer.add("prefix");
    buffer << ":";
    buffer.add(uint32_t(0x1a2b3c4d));
    buffer << ":";
    buffer.add(uint64_t(0x0123456789abcdefull));

    CHECK_EQ(std::string(buffer.view()), "prefix:1a2b3c4d:0123456789abcdef");
}

TEST_SUITE_END();
