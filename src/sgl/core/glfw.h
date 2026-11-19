// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/config.h"
#include "sgl/core/macros.h"

#if SGL_HAS_GLFW

namespace sgl::detail {

// Initialize on first use, on the main thread. Failed initialization can be retried.
void ensure_glfw_initialized();

// Release native windows and terminate GLFW on the main thread. Idempotent;
// further initialization is rejected. Also called by the Python exit handler.
void shutdown_glfw();

} // namespace sgl::detail

#endif
