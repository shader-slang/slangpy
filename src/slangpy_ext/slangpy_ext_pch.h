// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// STL headers
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

// Third-party headers
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4061)
#pragma warning(disable : 4459)
#endif
#include <fmt/format.h>
#include <fmt/ranges.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// Nanobind (heavy template machinery)
#include "nanobind.h"
