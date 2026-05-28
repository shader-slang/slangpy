// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "string.h"
#include "sgl/core/error.h"
#include "sgl/core/format.h"

#include <algorithm>
#include <cctype>

namespace sgl::string {

void copy_to_cstr(char* dst, size_t dst_len, std::string_view src)
{
    size_t len = std::min(dst_len - 1, src.size());
    std::memcpy(dst, src.data(), len);
    dst[len] = '\0';
}

std::string to_lower(std::string_view str)
{
    std::string result(str.size(), 0);
    for (size_t i = 0; i < str.size(); ++i)
        result[i] = char(::tolower(str[i]));
    return result;
}

std::string to_upper(std::string_view str)
{
    std::string result(str.size(), 0);
    for (size_t i = 0; i < str.size(); ++i)
        result[i] = char(::toupper(str[i]));
    return result;
}

bool has_prefix(std::string_view str, std::string_view prefix, bool case_sensitive)
{
    if (case_sensitive)
        return str.starts_with(prefix);
    else
        return to_lower(str).starts_with(to_lower(prefix));
}

bool has_suffix(std::string_view str, std::string_view suffix, bool case_sensitive)
{
    if (case_sensitive)
        return str.ends_with(suffix);
    else
        return to_lower(str).ends_with(to_lower(suffix));
}

std::vector<std::string> split(std::string_view str, std::string_view delimiters)
{
    std::string token;
    std::vector<std::string> result;

    for (const char c : str) {
        if (delimiters.find(c) != std::string::npos) {
            if (!token.empty()) {
                result.push_back(std::move(token));
                token.clear();
            }
        } else {
            token += c;
        }
    }

    if (token.length())
        result.push_back(std::move(token));

    return result;
}

std::string join(std::span<const std::string> strings, std::string_view separator)
{
    std::string result;
    for (auto it = strings.begin(); it != strings.end(); it++) {
        result += *it;
        if (it != strings.end() - 1)
            result += separator;
    }
    return result;
}

std::string indent(std::string_view str, std::string_view indentation)
{
    std::string result;
    for (auto c : str) {
        result += c;
        if (c == '\n')
            result += indentation;
    }
    return result;
}

std::string remove_leading_whitespace(std::string_view str, std::string_view whitespace)
{
    std::string result(str);
    result.erase(0, result.find_first_not_of(whitespace));
    return result;
}

std::string remove_trailing_whitespace(std::string_view str, std::string_view whitespace)
{
    std::string result(str);
    result.erase(result.find_last_not_of(whitespace) + 1);
    return result;
}

std::string remove_leading_trailing_whitespace(std::string_view str, std::string_view whitespace)
{
    return remove_trailing_whitespace(remove_leading_whitespace(str, whitespace), whitespace);
}

std::string format_byte_size(size_t size)
{
    if (size < 1024ull)
        return fmt::format("{} B", size);
    else if (size < 1048576ull)
        return fmt::format("{:.2f} kB", size / 1024.0);
    else if (size < 1073741824ull)
        return fmt::format("{:.2f} MB", size / 1048576.0);
    else if (size < 1099511627776ull)
        return fmt::format("{:.2f} GB", size / 1073741824.0);
    else
        return fmt::format("{:.2f} TB", size / 1099511627776.0);
}

std::string format_duration(double seconds)
{
    if (seconds < 1e-6)
        return fmt::format("{:.2f} ns", seconds * 1e9);
    else if (seconds < 1e-3)
        return fmt::format("{:.2f} us", seconds * 1e6);
    else if (seconds < 1.0)
        return fmt::format("{:.2f} ms", seconds * 1e3);
    else if (seconds < 60.0)
        return fmt::format("{:.2f} s", seconds);
    else if (seconds < 3600.0)
        return fmt::format("{:.2f} m", seconds / 60);
    else if (seconds < 86400.0)
        return fmt::format("{:.2f} h", seconds / 3600);
    else
        return fmt::format("{:.2f} d", seconds / 86400);
}

std::string hexlify(const void* data, size_t len)
{
    static const char* HEX_CHARS = "0123456789abcdef";
    std::string result(len * 2, 0);
    for (size_t i = 0; i < len; ++i) {
        uint8_t b = reinterpret_cast<const uint8_t*>(data)[i];
        result[i * 2] = HEX_CHARS[b >> 4];
        result[i * 2 + 1] = HEX_CHARS[b & 0xf];
    }
    return result;
}

std::string encode_base64(const void* data, size_t len)
{
    // based on https://gist.github.com/tomykaira/f0fd86b6c73063283afe550bc5d77594
    // clang-format off
    static constexpr char k_encoding_table[] = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
        'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
        'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
        'w', 'x', 'y', 'z', '0', '1', '2', '3',
        '4', '5', '6', '7', '8', '9', '+', '/'
    };
    // clang-format on

    size_t out_len = 4 * ((len + 2) / 3);
    std::string out(out_len, '\0');

    const uint8_t* src = reinterpret_cast<const uint8_t*>(data);
    auto dst = out.data();

    size_t i;
    for (i = 0; i + 2 < len; i += 3) {
        *dst++ = k_encoding_table[(src[i] >> 2) & 0x3f];
        *dst++ = k_encoding_table[((src[i] & 0x3) << 4) | ((src[i + 1] & 0xf0) >> 4)];
        *dst++ = k_encoding_table[((src[i + 1] & 0xf) << 2) | ((src[i + 2] & 0xc0) >> 6)];
        *dst++ = k_encoding_table[src[i + 2] & 0x3f];
    }
    if (i < len) {
        *dst++ = k_encoding_table[(src[i] >> 2) & 0x3f];
        if (i == (len - 1)) {
            *dst++ = k_encoding_table[((src[i] & 0x3) << 4)];
            *dst++ = '=';
        } else {
            *dst++ = k_encoding_table[((src[i] & 0x3) << 4) | ((src[i + 1] & 0xf0) >> 4)];
            *dst++ = k_encoding_table[((src[i + 1] & 0xf) << 2)];
        }
        *dst++ = '=';
    }

    return out;
}

std::vector<uint8_t> decode_base64(std::string_view str)
{
    // based on https://gist.github.com/tomykaira/f0fd86b6c73063283afe550bc5d77594
    // clang-format off
    static constexpr uint8_t k_decoding_table[] = {
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
        64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
        64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
    };
    // clang-format on

    size_t in_len = str.size();
    if (in_len == 0)
        return {};
    if (in_len % 4 != 0)
        SGL_THROW("Input data size is not a multiple of 4");

    size_t out_len = in_len / 4 * 3;
    if (str[in_len - 1] == '=')
        out_len--;
    if (str[in_len - 2] == '=')
        out_len--;

    std::vector<uint8_t> out(out_len, 0);

    for (size_t i = 0, j = 0; i < in_len;) {
        uint32_t a = str[i] == '=' ? 0 & i++ : k_decoding_table[static_cast<uint32_t>(str[i++])];
        uint32_t b = str[i] == '=' ? 0 & i++ : k_decoding_table[static_cast<uint32_t>(str[i++])];
        uint32_t c = str[i] == '=' ? 0 & i++ : k_decoding_table[static_cast<uint32_t>(str[i++])];
        uint32_t d = str[i] == '=' ? 0 & i++ : k_decoding_table[static_cast<uint32_t>(str[i++])];

        uint32_t triple = (a << 3 * 6) + (b << 2 * 6) + (c << 1 * 6) + (d << 0 * 6);

        if (j < out_len)
            out[j++] = (triple >> 2 * 8) & 0xff;
        if (j < out_len)
            out[j++] = (triple >> 1 * 8) & 0xff;
        if (j < out_len)
            out[j++] = (triple >> 0 * 8) & 0xff;
    }

    return out;
}

} // namespace sgl::string
