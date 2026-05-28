// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/error.h"

#include <utility>

namespace sgl {

template<typename R, typename T>
R narrow_cast(T value)
{
    if (!std::in_range<R>(value))
        SGL_THROW("narrow_cast failed");
    return static_cast<R>(value);
}

} // namespace sgl
