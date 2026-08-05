// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

namespace sgl {
class Profiler;
}

namespace sgl::ui {

/// Render real-time frame-aligned profiler statistics from cached snapshots.
/// @param profiler Profiler to display, or nullptr to use the current profiler.
SGL_API void render_profiler_window(Profiler* profiler = nullptr);

} // namespace sgl::ui
