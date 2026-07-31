// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/core/memory_stream.h"
#include "sgl/core/platform.h"
#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/utils/texture_loader.h"

#include <cstring>
#include <fstream>
#include <vector>

using namespace sgl;

TEST_SUITE_BEGIN("device");

namespace {

std::vector<uint8_t> read_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    REQUIRE(stream);
    const auto size = stream.tellg();
    REQUIRE(size > 0);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    REQUIRE(stream);
    return data;
}

void check_stream_matches_path(testing::GpuTestContext& ctx, const std::filesystem::path& path)
{
    const std::vector<uint8_t> data = read_file(path);
    MemoryStream stream(data.data(), data.size());
    TextureLoader loader{ref<Device>(ctx.device)};

    ref<Texture> path_texture = loader.load_texture(path);
    ref<Texture> stream_texture = loader.load_texture(&stream);

    CHECK(stream_texture->type() == path_texture->type());
    CHECK(stream_texture->format() == path_texture->format());
    CHECK(stream_texture->width() == path_texture->width());
    CHECK(stream_texture->height() == path_texture->height());
    CHECK(stream_texture->mip_count() == path_texture->mip_count());
    CHECK(stream_texture->array_length() == path_texture->array_length());

    for (uint32_t layer = 0; layer < path_texture->layer_count(); ++layer) {
        for (uint32_t mip = 0; mip < path_texture->mip_count(); ++mip) {
            const OwnedSubresourceData path_data = path_texture->get_subresource_data(layer, mip);
            const OwnedSubresourceData stream_data = stream_texture->get_subresource_data(layer, mip);
            const SubresourceLayout packed_layout = path_texture->get_subresource_layout(mip, 1);

            REQUIRE(path_data.size == stream_data.size);
            REQUIRE(path_data.row_pitch == stream_data.row_pitch);
            REQUIRE(path_data.slice_pitch == stream_data.slice_pitch);

            for (uint32_t z = 0; z < packed_layout.size.z; ++z) {
                for (size_t row = 0; row < packed_layout.row_count; ++row) {
                    const auto* path_row = static_cast<const uint8_t*>(path_data.data) + z * path_data.slice_pitch
                        + row * path_data.row_pitch;
                    const auto* stream_row = static_cast<const uint8_t*>(stream_data.data) + z * stream_data.slice_pitch
                        + row * stream_data.row_pitch;
                    CHECK(std::memcmp(path_row, stream_row, packed_layout.row_pitch) == 0);
                }
            }
        }
    }
}

} // namespace

TEST_CASE_GPU("texture_loader_stream_bitmap")
{
    const std::filesystem::path path = platform::project_directory() / "data" / "test_images" / "albert.jpg";
    check_stream_matches_path(ctx, path);
}

TEST_CASE_GPU("texture_loader_stream_dds")
{
    const std::filesystem::path path = platform::project_directory() / "data" / "test_images" / "dds" / "bc1-unorm.dds";
    check_stream_matches_path(ctx, path);
}

TEST_SUITE_END();
