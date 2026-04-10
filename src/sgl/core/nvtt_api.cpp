// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/core/nvtt_api.h"
#include "sgl/core/error.h"
#include "sgl/core/logger.h"

#include <cstdlib>
#include <mutex>

namespace sgl {

//
// NvttAPI dynamic loading
//

void NvttAPI::try_load()
{
    // 1. Try runtime_directory() / library name.
#if SGL_WINDOWS
    const char* lib_name = "nvtt.dll";
#elif SGL_LINUX
    const char* lib_name = "libnvtt.so";
#elif SGL_MACOS
    const char* lib_name = "libnvtt.dylib";
#else
    const char* lib_name = nullptr;
#endif

    if (!lib_name) {
        log_info("NVTT3 not supported on this platform.");
        return;
    }

    // Try next to sgl library.
    if (load_from_path(platform::runtime_directory() / lib_name))
        goto resolve;

    // 2. Try bare name (system PATH / LD_LIBRARY_PATH).
    if (load_from_path(lib_name))
        goto resolve;

    // 3. Try SGL_NVTT_PATH environment variable.
    {
#ifdef _MSC_VER
        char* env_path = nullptr;
        size_t env_len = 0;
        _dupenv_s(&env_path, &env_len, "SGL_NVTT_PATH");
        if (env_path && env_path[0] != '\0') {
            bool ok = load_from_path(std::filesystem::path(env_path) / lib_name);
            free(env_path);
            if (ok)
                goto resolve;
        } else {
            free(env_path);
        }
#else
        const char* env_path = std::getenv("SGL_NVTT_PATH");
        if (env_path && env_path[0] != '\0') {
            if (load_from_path(std::filesystem::path(env_path) / lib_name))
                goto resolve;
        }
#endif
    }

    log_info("NVTT3 not found, BC6H encoding unavailable.");
    return;

resolve:
    if (!resolve_symbols()) {
        log_warn("NVTT3 library found but symbols could not be resolved.");
        platform::release_shared_library(library);
        library = nullptr;
        available = false;
        return;
    }

    unsigned int version = nvttVersion();
    log_info("NVTT3 loaded (version {}.{}.{}).", (version >> 16) & 0xff, (version >> 8) & 0xff, version & 0xff);
    available = true;
}

bool NvttAPI::load_from_path(const std::filesystem::path& path)
{
    library = platform::load_shared_library(path);
    return library != nullptr;
}

bool NvttAPI::resolve_symbols()
{
#define NVTT_RESOLVE(name)                                                                                             \
    name = reinterpret_cast<decltype(name)>(platform::get_proc_address(library, #name));                               \
    if (!name)                                                                                                         \
        return false;

    NVTT_RESOLVE(nvttCreateCPUInputBuffer);
    NVTT_RESOLVE(nvttDestroyCPUInputBuffer);
    NVTT_RESOLVE(nvttEncodeCPU);
    NVTT_RESOLVE(nvttCreateSurface);
    NVTT_RESOLVE(nvttDestroySurface);
    NVTT_RESOLVE(nvttSurfaceSetImageData);
    NVTT_RESOLVE(nvttSurfaceBuildNextMipmapDefaults);
    NVTT_RESOLVE(nvttSurfaceData);
    NVTT_RESOLVE(nvttSurfaceWidth);
    NVTT_RESOLVE(nvttSurfaceHeight);
    NVTT_RESOLVE(nvttVersion);

#undef NVTT_RESOLVE
    return true;
}

//
// Singleton access
//

NvttAPI& get_nvtt_api()
{
    static NvttAPI api;
    static std::once_flag flag;
    std::call_once(
        flag,
        [&]
        {
            api.try_load();
        }
    );
    return api;
}

} // namespace sgl
