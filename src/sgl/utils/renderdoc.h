// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/fwd.h"
#include "sgl/device/fwd.h"

namespace sgl::renderdoc {

/**
 * \brief Check if RenderDoc is available.
 *
 * This is typically the case when the application is running under the RenderDoc.
 *
 * \return True if RenderDoc is available.
 */
SGL_API bool is_available();

/**
 * \brief Start capturing a frame in RenderDoc.
 *
 * This function will start capturing a frame (or some partial compute/graphics workload) in RenderDoc.
 *
 * To end the frame capture, call \c end_frame_capture().
 *
 * \param device The device to capture the frame for.
 * \param window The window to capture the frame for (optional).
 *
 * \return True if the frame capture was started successfully.
 */
SGL_API bool start_frame_capture(ref<Device> device, ref<Window> window = nullptr);

/**
 * \brief End capturing a frame in RenderDoc.
 *
 * This function will end capturing a frame (or some partial compute/graphics workload) in RenderDoc.
 *
 * \return True if the frame capture was ended successfully.
 */
SGL_API bool end_frame_capture();

/**
 * \brief Check if a frame is currently being captured in RenderDoc.
 *
 * \return True if a frame is currently being captured.
 */
SGL_API bool is_frame_capturing();

} // namespace sgl::renderdoc
