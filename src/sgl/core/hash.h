// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <utility>
#include <functional>

namespace sgl {

inline size_t hash_combine(size_t hash1, size_t hash2)
{
    return hash2 ^ (hash1 + 0x9e3779b9 + (hash2 << 6) + (hash2 >> 2));
}

template<typename T>
size_t hash(const T& t)
{
    return std::hash<T>()(t);
}

template<typename T1, typename T2, typename... Rest>
size_t hash(const T1& t1, const T2& t2, const Rest&... rest)
{
    size_t result = hash_combine(hash(t1), hash(t2));
    ((result = hash_combine(result, hash(rest))), ...);
    return result;
}

template<typename T1, typename T2>
size_t hash(const std::pair<T1, T2>& t)
{
    return hash_combine(hash(t.first), hash(t.second));
}

template<typename T>
struct hasher {
    size_t operator()(const T& t) const { return hash(t); }
};

template<typename T>
struct comparator {
    size_t operator()(const T& t1, const T& t2) const { return t1 == t2; }
};

} // namespace sgl
