// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "crashpad.h"

#include "sgl/core/config.h"
#include "sgl/core/error.h"

#if SGL_HAS_CRASHPAD

#include "sgl/core/platform.h"

#include <crashpad/client/crashpad_client.h>
#include <base/logging.h>

namespace sgl::crashpad {

bool is_supported()
{
    return true;
}

static ::crashpad::CrashpadClient* s_client;

void start_handler(
    std::filesystem::path handler,
    std::filesystem::path database,
    std::map<std::string, std::string> annotations
)
{
    if (s_client)
        SGL_THROW("Crashpad handler has already been started.");

    if (handler.empty()) {
#if SGL_WINDOWS
        handler = platform::runtime_directory() / "crashpad_handler.exe";
#else
        handler = platform::runtime_directory() / "crashpad_handler";
#endif
    }

    if (database.empty())
        database = platform::runtime_directory() / "crashpad_database";

    std::filesystem::create_directories(database);

    ::logging::InitLogging({.logging_dest = ::logging::LOG_TO_STDERR});

    s_client = new ::crashpad::CrashpadClient();

    bool success = s_client->StartHandler(
        ::base::FilePath{handler.native()},
        ::base::FilePath{database.native()},
        {}, // metrics_dir
        {}, // url
        annotations,
        {},    // arguments,
        false, // restartable
        false  // asynchronous_start
    );
    if (!success) {
        delete s_client;
        s_client = nullptr;
        SGL_THROW("Failed to start Crashpad handler.");
    }
}

} // namespace sgl::crashpad

#else // SGL_HAS_CRASHPAD

namespace sgl::crashpad {

bool is_supported()
{
    return false;
}

void start_handler(
    std::filesystem::path handler,
    std::filesystem::path database,
    std::map<std::string, std::string> annotations
)
{
    SGL_UNUSED(handler, database, annotations);
    SGL_THROW("Crashpad is not supported in this build.\nEnable it using the SGL_ENABLE_CRASHPAD cmake option.");
}

} // namespace sgl::crashpad

#endif // SGL_HAS_CRASHPAD
