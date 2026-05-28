// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <type_traits>

namespace sgl {

template<typename T>
struct is_boolean : std::is_same<T, bool> { };

template<typename T>
inline constexpr bool is_boolean_v = is_boolean<T>::value;

} // namespace sgl
