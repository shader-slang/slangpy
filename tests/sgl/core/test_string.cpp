// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/string.h"

using namespace sgl;

TEST_SUITE_BEGIN("string");

TEST_CASE("copy_to_cstr")
{
    char cstr[16];
    string::copy_to_cstr(cstr, sizeof(cstr), "");
    CHECK_EQ(std::string_view(cstr), "");
    string::copy_to_cstr(cstr, sizeof(cstr), "hello");
    CHECK_EQ(std::string_view(cstr), "hello");
    string::copy_to_cstr(cstr, sizeof(cstr), "this string will be truncated");
    CHECK_EQ(std::string_view(cstr), "this string wil");
}

TEST_CASE("to_upper")
{
    CHECK_EQ(string::to_upper("hello"), "HELLO");
    CHECK_EQ(string::to_upper("HELLO"), "HELLO");
    CHECK_EQ(string::to_upper("Hello"), "HELLO");
    CHECK_EQ(string::to_upper("hElLo"), "HELLO");
}

TEST_CASE("to_lower")
{
    CHECK_EQ(string::to_lower("hello"), "hello");
    CHECK_EQ(string::to_lower("HELLO"), "hello");
    CHECK_EQ(string::to_lower("Hello"), "hello");
    CHECK_EQ(string::to_lower("hElLo"), "hello");
}

TEST_CASE("has_prefix")
{
    CHECK(string::has_prefix("hello", "he", true));
    CHECK(string::has_prefix("hello", "He", false));
    CHECK(string::has_prefix("hello", "HELLO", false));
    CHECK(!string::has_prefix("hello", "HE", true));
    CHECK(string::has_prefix("hello", "HE", false));
}

TEST_CASE("has_suffix")
{
    CHECK(string::has_suffix("hello", "lo", true));
    CHECK(string::has_suffix("hello", "Lo", false));
    CHECK(string::has_suffix("hello", "HELLO", false));
    CHECK(!string::has_suffix("hello", "LO", true));
    CHECK(string::has_suffix("hello", "LO", false));
}

TEST_CASE("split")
{
    CHECK_EQ(string::split("hello world", " "), (std::vector<std::string>{"hello", "world"}));
    CHECK_EQ(string::split("hello world", "o"), (std::vector<std::string>{"hell", " w", "rld"}));
    CHECK_EQ(string::split("hello world", "e"), (std::vector<std::string>{"h", "llo world"}));
    CHECK_EQ(string::split("hello world", " "), (std::vector<std::string>{"hello", "world"}));
    CHECK_EQ(string::split("hello world", "o"), (std::vector<std::string>{"hell", " w", "rld"}));
    CHECK_EQ(string::split("hello world", "e"), (std::vector<std::string>{"h", "llo world"}));
}

TEST_CASE("join")
{
    CHECK_EQ(string::join(std::vector<std::string>{"hello", "world"}, " "), "hello world");
    CHECK_EQ(string::join(std::vector<std::string>{"hello", "world"}, "o"), "hellooworld");
    CHECK_EQ(string::join(std::vector<std::string>{"hello", "world"}, "e"), "helloeworld");
    CHECK_EQ(string::join(std::vector<std::string>{"hello", "world"}, " "), "hello world");
    CHECK_EQ(string::join(std::vector<std::string>{"hello", "world"}, "o"), "hellooworld");
    CHECK_EQ(string::join(std::vector<std::string>{"hello", "world"}, "e"), "helloeworld");
}

TEST_CASE("remove_whitespace")
{
    const char* whitespace = " \t\n\r";
    CHECK_EQ(string::remove_leading_whitespace("  \t\t\n\n\r\rtest", whitespace), "test");
    CHECK_EQ(string::remove_leading_whitespace("test", whitespace), "test");
    CHECK_EQ(string::remove_leading_whitespace("test  \t\t\n\n\r\r", whitespace), "test  \t\t\n\n\r\r");

    CHECK_EQ(string::remove_trailing_whitespace("  \t\t\n\n\r\rtest", whitespace), "  \t\t\n\n\r\rtest");
    CHECK_EQ(string::remove_trailing_whitespace("test", whitespace), "test");
    CHECK_EQ(string::remove_trailing_whitespace("test  \t\t\n\n\r\r", whitespace), "test");

    CHECK_EQ(string::remove_leading_trailing_whitespace("  \t\t\n\n\r\rtest", whitespace), "test");
    CHECK_EQ(string::remove_leading_trailing_whitespace("test", whitespace), "test");
    CHECK_EQ(string::remove_leading_trailing_whitespace("test  \t\t\n\n\r\r", whitespace), "test");
}

TEST_CASE("format_byte_size")
{
    const size_t kB = 1024ull;
    const size_t MB = 1048576ull;
    const size_t GB = 1073741824ull;
    const size_t TB = 1099511627776ull;

    CHECK_EQ(string::format_byte_size(0), "0 B");
    CHECK_EQ(string::format_byte_size(100), "100 B");
    CHECK_EQ(string::format_byte_size(1023), "1023 B");
    CHECK_EQ(string::format_byte_size(kB), "1.00 kB");
    CHECK_EQ(string::format_byte_size(100 * kB), "100.00 kB");
    CHECK_EQ(string::format_byte_size(1023 * kB), "1023.00 kB");
    CHECK_EQ(string::format_byte_size(MB), "1.00 MB");
    CHECK_EQ(string::format_byte_size(10 * MB), "10.00 MB");
    CHECK_EQ(string::format_byte_size(1023 * MB), "1023.00 MB");
    CHECK_EQ(string::format_byte_size(GB), "1.00 GB");
    CHECK_EQ(string::format_byte_size(10 * GB), "10.00 GB");
    CHECK_EQ(string::format_byte_size(1023 * GB), "1023.00 GB");
    CHECK_EQ(string::format_byte_size(TB), "1.00 TB");
    CHECK_EQ(string::format_byte_size(10 * TB), "10.00 TB");
}

TEST_CASE("format_duration")
{
    CHECK_EQ(string::format_duration(0.0), "0.00 ns");
    CHECK_EQ(string::format_duration(1e-9), "1.00 ns");
    CHECK_EQ(string::format_duration(1e-6), "1.00 us");
    CHECK_EQ(string::format_duration(1e-3), "1.00 ms");
    CHECK_EQ(string::format_duration(1.0), "1.00 s");
    CHECK_EQ(string::format_duration(60.0), "1.00 m");
    CHECK_EQ(string::format_duration(3600.0), "1.00 h");
    CHECK_EQ(string::format_duration(86400.0), "1.00 d");
}

TEST_CASE("base64")
{
    auto test_encode_decode = [&](std::string decoded, std::string encoded)
    {
        CHECK_EQ(string::encode_base64(std::vector<uint8_t>(decoded.begin(), decoded.end())), encoded);
        CHECK_EQ(string::decode_base64(encoded), std::vector<uint8_t>(decoded.begin(), decoded.end()));
    };

    test_encode_decode("", "");
    test_encode_decode("a", "YQ==");
    test_encode_decode("ab", "YWI=");
    test_encode_decode("abc", "YWJj");
    test_encode_decode("Hello World!", "SGVsbG8gV29ybGQh");
    test_encode_decode(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore "
        "magna aliqua.",
        "TG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQsIGNvbnNlY3RldHVyIGFkaXBpc2NpbmcgZWxpdCwgc2VkIGRvIGVpdXNtb2QgdGVtcG9yIGluY2"
        "lkaWR1bnQgdXQgbGFib3"
        "JlIGV0IGRvbG9yZSBtYWduYSBhbGlxdWEu"
    );
}

TEST_SUITE_END();
