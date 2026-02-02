// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#include <filesystem>
#include <map>
#include <optional>
#include <string>

namespace sgl::crashpad {

/// Returns true if Crashpad is supported in this build.
SGL_API bool is_supported();

/**
 * \brief Starts the Crashpad handler.
 *
 * Start the chromium Crashpad handler to capture crashes and generate crash reports.
 *
 * \param handler Path to the handler executable. Defaults to `<runtime_directory>/crashpad_handler{.exe}` if empty.
 * \param database Path to the database directory. Defaults to `<runtime_directory>/crashpad_database` if empty.
 * \param annotations Annotations to include with crash reports.
 */
SGL_API void start_handler(
    std::filesystem::path handler = {},
    std::filesystem::path database = {},
    std::map<std::string, std::string> annotations = {}
);

} // namespace sgl::crashpad
