// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/device/device.h"
#include "sgl/device/detail/profile.h"

using namespace sgl;

TEST_SUITE_BEGIN("profile");

TEST_CASE("parse")
{
    CHECK_EQ(parse_target_profile("sm_6_10"), TargetProfile{TargetProfileFamily::shader_model, 6, 10});
    CHECK_EQ(parse_target_profile("spirv_1_6"), TargetProfile{TargetProfileFamily::spirv, 1, 6});
    CHECK_EQ(parse_target_profile("metallib_3_2"), TargetProfile{TargetProfileFamily::metallib, 3, 2});
    CHECK_FALSE(parse_target_profile("cuda_sm_9_0"));
    CHECK_FALSE(parse_target_profile("sm_6"));
    CHECK_FALSE(parse_target_profile("sm_6_1_extra"));
}

TEST_CASE("normalize-capability")
{
    CHECK_EQ(target_profile_from_capability("_sm_6_10"), TargetProfile{TargetProfileFamily::shader_model, 6, 10});
    CHECK_EQ(target_profile_from_capability("_spirv_1_6"), TargetProfile{TargetProfileFamily::spirv, 1, 6});
    CHECK_EQ(target_profile_from_capability("metallib_4_0"), TargetProfile{TargetProfileFamily::metallib, 4, 0});
    CHECK_FALSE(target_profile_from_capability("_cuda_sm_9_0"));
}

TEST_CASE("build-inventory")
{
    const std::vector<std::string> candidates = {
        "_spirv_1_6",
        "_sm_6_10",
        "sm_6_9",
        "spirv_1_5",
        "sm_6_10",
        "metallib_3_2",
        "invalid",
    };
    CHECK_EQ(
        build_target_profile_inventory(
            candidates,
            DeviceType::vulkan,
            [](std::string_view profile)
            {
                return profile != "sm_6_10";
            }
        ),
        std::vector<std::string>{"sm_6_9", "spirv_1_5", "spirv_1_6"}
    );
    CHECK_EQ(
        build_target_profile_inventory(
            candidates,
            DeviceType::d3d12,
            [](std::string_view)
            {
                return true;
            }
        ),
        std::vector<std::string>{"sm_6_9", "sm_6_10"}
    );
    CHECK_EQ(
        build_target_profile_inventory(
            candidates,
            DeviceType::metal,
            [](std::string_view)
            {
                return true;
            }
        ),
        std::vector<std::string>{"metallib_3_2"}
    );
    CHECK_EQ(
        build_target_profile_inventory(
            candidates,
            DeviceType::cuda,
            [](std::string_view)
            {
                return true;
            }
        ),
        std::vector<std::string>{}
    );
    CHECK_EQ(
        build_target_profile_inventory(
            candidates,
            DeviceType::vulkan,
            [](std::string_view)
            {
                return true;
            }
        ),
        std::vector<std::string>{"sm_6_9", "sm_6_10", "spirv_1_5", "spirv_1_6"}
    );
}

TEST_CASE("highest-family-profile")
{
    const std::vector<std::string> profiles = {
        "sm_6_9",
        "sm_6_10",
        "spirv_1_5",
        "spirv_1_6",
    };
    CHECK_EQ(highest_target_profile(profiles, TargetProfileFamily::shader_model), "sm_6_10");
    CHECK_EQ(highest_target_profile(profiles, TargetProfileFamily::spirv), "spirv_1_6");
    CHECK_EQ(highest_target_profile(profiles, TargetProfileFamily::metallib), "");
}

TEST_SUITE_END();
