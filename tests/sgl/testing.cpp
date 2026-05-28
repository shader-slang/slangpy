// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/core/macros.h"

#include "sgl/device/device.h"

#include <map>
#include <ctime>

namespace sgl::testing {

// Cached devices.
static std::map<DeviceType, ref<Device>> g_cached_devices;

// Temp directory to create files for teting in.
static std::filesystem::path g_test_temp_directory;

// Calculates a files sytem compatible date string formatted YYYY-MM-DD-hh-mm-ss.
static std::string build_current_date_string()
{
    time_t now;
    time(&now);
    struct tm tm;
#if SGL_WINDOWS
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char result[128];
    std::strftime(result, sizeof(result), "%Y-%m-%d-%H-%M-%S", &tm);
    return result;
}

std::filesystem::path get_test_temp_directory()
{
    if (g_test_temp_directory == "") {
        std::string datetime_str = build_current_date_string();
        g_test_temp_directory = std::filesystem::current_path() / ".test_temp" / datetime_str;
        std::filesystem::create_directories(g_test_temp_directory);
    }
    return g_test_temp_directory;
}

std::filesystem::path get_suite_temp_directory()
{
    auto path = get_test_temp_directory() / get_current_test_suite_name();
    std::filesystem::create_directories(path);
    return path;
}

std::filesystem::path get_case_temp_directory()
{
    auto path = get_test_temp_directory() / get_current_test_suite_name() / get_current_test_case_name();
    std::filesystem::create_directories(path);
    return path;
}

void static_init() { }

void static_shutdown()
{
    // Clean up temp files
    std::filesystem::remove_all(g_test_temp_directory);

    g_cached_devices.clear();
}

void run_gpu_test(void (*func)(GpuTestContext&))
{
#if SGL_WINDOWS
    std::vector<DeviceType> device_types{DeviceType::d3d12, DeviceType::vulkan};
#elif SGL_LINUX
    std::vector<DeviceType> device_types{DeviceType::vulkan};
#elif SGL_MACOS
    std::vector<DeviceType> device_types{DeviceType::metal};
#endif

    bool use_cached_device = true;

    for (DeviceType device_type : device_types) {
        SUBCASE(enum_to_string(device_type).c_str())
        {
            ref<Device> device;
            if (use_cached_device) {
                auto it = g_cached_devices.find(device_type);
                if (it != g_cached_devices.end())
                    device = it->second;
            }

            if (!device) {
                DeviceDesc desc{
                    .type = device_type,
                    .enable_debug_layers = true,
                };
                device = Device::create(desc);
                if (use_cached_device)
                    g_cached_devices[device_type] = device;
            }

            GpuTestContext ctx{.device = device};
            func(ctx);

            if (!use_cached_device)
                device->close();
        }
    }
}

} // namespace sgl::testing
