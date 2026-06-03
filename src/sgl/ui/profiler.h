// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"

#include "sgl/core/macros.h"

namespace sgl::ui {

/// Render an ImGui profiler window for a profiler stats snapshot.
SGL_API void render_profiler_window(Profiler* profiler = nullptr);

} // namespace sgl::ui
