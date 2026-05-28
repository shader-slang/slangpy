// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/fwd.h"

#include "sgl/device/fwd.h"

#include <string>
#include <cstdint>

namespace sgl::tev {

/**
 * \brief Show a bitmap in the tev viewer (https://github.com/Tom94/tev).
 *
 * This will block until the image is sent over.
 *
 * \param bitmap Bitmap to show.
 * \param name Name of the image in tev. If not specified, a unique name will be generated.
 * \param host Host to connect to.
 * \param port Port to connect to.
 * \param max_retries Maximum number of retries.
 * \return True if successful.
 */
SGL_API bool show(
    const Bitmap* bitmap,
    std::string name = "",
    std::string host = "127.0.0.1",
    uint16_t port = 14158,
    uint32_t max_retries = 3
);

/**
 * \brief Show texture in the tev viewer (https://github.com/Tom94/tev).
 *
 * This will block until the image is sent over.
 *
 * \param texture Texture to show.
 * \param name Name of the image in tev. If not specified, a unique name will be generated.
 * \param host Host to connect to.
 * \param port Port to connect to.
 * \param max_retries Maximum number of retries.
 * \return True if successful.
 */
SGL_API bool show(
    const Texture* texture,
    std::string name = "",
    std::string host = "127.0.0.1",
    uint16_t port = 14158,
    uint32_t max_retries = 3
);

/**
 * \brief Show a bitmap in the tev viewer (https://github.com/Tom94/tev).
 *
 * This will return immediately and send the image asynchronously in the background.
 *
 * \param bitmap Bitmap to show.
 * \param name Name of the image in tev. If not specified, a unique name will be generated.
 * \param host Host to connect to.
 * \param port Port to connect to.
 * \param max_retries Maximum number of retries.
 */
SGL_API void show_async(
    const Bitmap* bitmap,
    std::string name = "",
    std::string host = "127.0.0.1",
    uint16_t port = 14158,
    uint32_t max_retries = 3
);

/**
 * \brief Show a texture in the tev viewer (https://github.com/Tom94/tev).
 *
 * This will return immediately and send the image asynchronously in the background.
 *
 * \param bitmap Texture to show.
 * \param name Name of the image in tev. If not specified, a unique name will be generated.
 * \param host Host to connect to.
 * \param port Port to connect to.
 * \param max_retries Maximum number of retries.
 */
SGL_API void show_async(
    const Texture* texture,
    std::string name = "",
    std::string host = "127.0.0.1",
    uint16_t port = 14158,
    uint32_t max_retries = 3
);

} // namespace sgl::tev
