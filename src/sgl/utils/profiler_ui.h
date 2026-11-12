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

/// Reset window state and release cached snapshots. Call before Python finalization
/// so snapshots shared with Python are released while the interpreter is still usable.
SGL_API void reset_profiler_window();

} // namespace sgl::ui
