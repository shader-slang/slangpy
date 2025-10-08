// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"

#include <span>
#include <string_view>
#include <string>
#include <vector>

namespace sgl::string {

/**
 * Copy a string to a C-style string buffer.
 * Truncates the input string if it doesn't fit the buffer and always null-terminates the buffer.
 * \param dst C-style string buffer.
 * \param dst_len Length of the buffer.
 * \param src Input string.
 */
SGL_API void copy_to_cstr(char* dst, size_t dst_len, std::string_view src);

/**
 * Convert a string to lower case.
 * \param str Input string.
 * \return Lower case string.
 */
[[nodiscard]] SGL_API std::string to_lower(std::string_view str);

/**
 * Convert a string to upper case.
 * \param str Input string.
 * \return Upper case string.
 */
[[nodiscard]] SGL_API std::string to_upper(std::string_view str);

/**
 * Convert a string to wide string.
 * \param str Input string.
 * \return Wide string.
 */
[[nodiscard]] inline std::wstring to_wstring(std::string_view str)
{
    return std::wstring(str.begin(), str.end());
}

// inline std::string to_string(std::wstring str)
// {
//     return std::string(str.begin(), str.end());
// }

/**
 * Check is a string has the specified prefix.
 * \param str Input string.
 * \param prefix Prefix to check for.
 * \param case_sensitive Whether comparison should be case-sensitive.
 * \return True if the input string has the specified prefix.
 */
[[nodiscard]] SGL_API bool has_prefix(std::string_view str, std::string_view prefix, bool case_sensitive = true);

/**
 * Check is a string has the specified suffix.
 * \param str Input string.
 * \param suffix Suffix to check for.
 * \param case_sensitive Whether comparison should be case-sensitive.
 * \return True if the input string has the specified suffix.
 */
[[nodiscard]] SGL_API bool has_suffix(std::string_view str, std::string_view suffix, bool case_sensitive = true);

/**
 * Split a string into a list of strings based on a list of delimiter characters.
 * \param str Input string.
 * \param delimiters Delimiter characters.
 * \return List of split strings excluding the delimiter characters.
 */
[[nodiscard]] SGL_API std::vector<std::string> split(std::string_view str, std::string_view delimiters);

/**
 * Join an list of strings separated by a specified seperator.
 * \param strings Input strings.
 * \param separator String placed between each string to be joined.
 * \return Joined string.
 */
[[nodiscard]] SGL_API std::string join(std::span<const std::string> strings, std::string_view separator);

/**
 * Replace every newline with the specified indentation.
 * \param str Input string.
 * \param indentation Indentation string.
 * \return The indented string.
 */
[[nodiscard]] SGL_API std::string indent(std::string_view str, std::string_view indentation = "  ");

template<typename T>
concept object_to_string = requires(typename std::remove_pointer<typename remove_ref<T>::type>::type t) {
    { t.to_string() } -> std::convertible_to<std::string>;
};

/**
 * Convert a list of objects that have a to_string() method to a string.
 * \tparam T Type of objects.
 * \param list List of objects.
 * \return The list of objects as a string.
 */
template<object_to_string T>
inline std::string list_to_string(std::span<T> list, std::string_view indentation = "  ")
{
    if (list.empty())
        return "[]";
    std::string result = "[\n";
    for (const auto& item : list) {
        result += indentation;
        if constexpr (std::is_pointer_v<T> || is_ref_v<T>)
            result += string::indent(item->to_string());
        else
            result += string::indent(item.to_string());
        result += ",\n";
    }
    result += "]";
    return result;
}

template<typename T>
concept free_standing_to_string = requires(T t) {
    { to_string(t) } -> std::convertible_to<std::string>;
};

/**
 * Convert a list of objects that have a free-standing to_string() method to a string.
 * \tparam T Type of objects.
 * \param list List of objects.
 * \return The list of objects as a string.
 */
template<free_standing_to_string T>
inline std::string list_to_string(std::span<T> list, std::string_view indentation = "  ")
{
    if (list.empty())
        return "[]";
    std::string result = "[\n";
    for (const auto& item : list) {
        result += indentation;
        result += string::indent(to_string(item));
        result += ",\n";
    }
    result += "]";
    return result;
}

template<typename T>
inline std::string list_to_string(const std::vector<T>& list, std::string_view indentation = "  ")
{
    return list_to_string(std::span{list}, indentation);
}

template<typename T>
concept iterable_has_to_string = requires(typename T::iterator t) {
    { (*t)->to_string() } -> std::convertible_to<std::string>;
};

template<iterable_has_to_string T>
inline std::string iterable_to_string(const T& iterable)
{
    std::string result = "[\n";
    for (const auto& item : iterable) {
        result += "  ";
        result += string::indent(item->to_string());
        result += ",\n";
    }
    result += "]";
    return result;
}

/**
 * Remove leading whitespace.
 * \param str Input string.
 * \param whitespace Whitespace characters.
 * \return String with leading whitespace removed.
 */
[[nodiscard]] SGL_API std::string
remove_leading_whitespace(std::string_view str, std::string_view whitespace = " \n\r\t");

/**
 * Remove trailing whitespace.
 * \param str Input string.
 * \param whitespace Whitespace characters.
 * \return String with trailing whitespace removed.
 */
[[nodiscard]] SGL_API std::string
remove_trailing_whitespace(std::string_view str, std::string_view whitespace = " \n\r\t");

/**
 * Remove leading and trailing whitespace.
 * \param str Input string.
 * \param whitespace Whitespace characters.
 * \return String with leading and trailing whitespace removed.
 */
[[nodiscard]] SGL_API std::string
remove_leading_trailing_whitespace(std::string_view str, std::string_view whitespace = " \n\r\t");

/**
 * Convert a size in bytes to a human readable string:
 * - prints bytes (B) if size < 1000 bytes
 * - prints kilobytes (KB) if size < 1000 kilobytes
 * - prints megabytes (MB) if size < 1000 megabytes
 * - prints gigabytes (GB) if size < 1000 gigabytes
 * - otherwise prints terabytes (TB)
 * \param size Size in bytes.
 * \return Size as a human readable string.
 */
[[nodiscard]] SGL_API std::string format_byte_size(size_t size);

/**
 * Convert a duration in seconds to a human readable string:
 * - prints nanoseconds (ns) if duration < 1 microsecond
 * - prints microseconds (us) if duration < 1 millisecond
 * - prints milliseconds (ms) if duration < 1 second
 * - prints seconds (s) if duration < 60 seconds
 * - prints minutes (m) if duration < 60 minutes
 * - prints hours (h) if duration < 24 hours
 * - otherwise prints days (d)
 * \param seconds Duration in seconds.
 * \return Duration as a human readable string.
 */
[[nodiscard]] SGL_API std::string format_duration(double seconds);

/**
 * Convert data to a hexadecimal string.
 * \param data Input data.
 * \param len Length of input data.
 * \return Hexadecimal string.
 */
[[nodiscard]] SGL_API std::string hexlify(const void* data, size_t len);

/**
 * Convert data to a hexadecimal string.
 * \param data Input data.
 * \return Hexadecimal string.
 */
[[nodiscard]] inline std::string hexlify(std::span<const uint8_t> data)
{
    return hexlify(data.data(), data.size());
}

/**
 * Encode data into base 64 encoding.
 * \param data Input data.
 * \param len Length of input data.
 * \return Base 64 encoded string.
 */
[[nodiscard]] SGL_API std::string encode_base64(const void* data, size_t len);

/**
 * Encode data into base 64 encoding.
 * \param data Input data.
 * \return Base 64 encoded string.
 */
[[nodiscard]] inline std::string encode_base64(std::span<const uint8_t> data)
{
    return encode_base64(data.data(), data.size());
}

/**
 * Decode data from base 64 encoding.
 * \param str Base 64 encoded string.
 * \return Decoded data.
 */
[[nodiscard]] SGL_API std::vector<uint8_t> decode_base64(std::string_view str);

} // namespace sgl::string
