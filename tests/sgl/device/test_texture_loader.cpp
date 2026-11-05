// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/core/bitmap.h"
#include "sgl/core/memory_stream.h"
#include "sgl/core/platform.h"
#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/utils/texture_loader.h"

#include <array>
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

void check_subresources_match(
    const Texture* actual,
    uint32_t actual_layer,
    const Texture* expected,
    uint32_t expected_layer,
    uint32_t mip
)
{
    const OwnedSubresourceData actual_data = actual->get_subresource_data(actual_layer, mip);
    const OwnedSubresourceData expected_data = expected->get_subresource_data(expected_layer, mip);
    const SubresourceLayout packed_layout = expected->get_subresource_layout(mip, 1);

    REQUIRE(actual_data.size == expected_data.size);
    REQUIRE(actual_data.row_pitch == expected_data.row_pitch);
    REQUIRE(actual_data.slice_pitch == expected_data.slice_pitch);

    for (uint32_t z = 0; z < packed_layout.size.z; ++z) {
        for (size_t row = 0; row < packed_layout.row_count; ++row) {
            const auto* actual_row = static_cast<const uint8_t*>(actual_data.data) + z * actual_data.slice_pitch
                + row * actual_data.row_pitch;
            const auto* expected_row = static_cast<const uint8_t*>(expected_data.data) + z * expected_data.slice_pitch
                + row * expected_data.row_pitch;
            CHECK(std::memcmp(actual_row, expected_row, packed_layout.row_pitch) == 0);
        }
    }
}

void check_textures_match(const Texture* actual, const Texture* expected)
{
    CHECK(actual->type() == expected->type());
    CHECK(actual->format() == expected->format());
    CHECK(actual->width() == expected->width());
    CHECK(actual->height() == expected->height());
    CHECK(actual->mip_count() == expected->mip_count());
    CHECK(actual->array_length() == expected->array_length());

    for (uint32_t layer = 0; layer < expected->layer_count(); ++layer) {
        for (uint32_t mip = 0; mip < expected->mip_count(); ++mip)
            check_subresources_match(actual, layer, expected, layer, mip);
    }
}

void check_stream_matches_path(testing::GpuTestContext& ctx, const std::filesystem::path& path)
{
    const std::vector<uint8_t> data = read_file(path);
    MemoryStream stream(data.data(), data.size());
    TextureLoader loader{ref<Device>(ctx.device)};

    ref<Texture> path_texture = loader.load_texture(path);
    ref<Texture> stream_texture = loader.load_texture(&stream);

    check_textures_match(stream_texture, path_texture);
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

TEST_CASE_GPU("texture_loader_encoded_images")
{
    const std::filesystem::path image_dir = platform::project_directory() / "data" / "test_images";
    const std::array paths{
        image_dir / "albert.jpg",
        image_dir / "png" / "uint8-4ch.png",
        image_dir / "dds" / "bc1-unorm.dds",
    };

    std::vector<std::vector<uint8_t>> bytes;
    std::vector<std::string> names;
    bytes.reserve(paths.size());
    names.reserve(paths.size());
    for (const auto& path : paths) {
        bytes.push_back(read_file(path));
        names.push_back(path.filename().string());
    }

    std::vector<TextureLoader::EncodedImageView> images;
    for (size_t i = 0; i < paths.size(); ++i)
        images.push_back({.data = std::span<const uint8_t>(bytes[i]), .name = names[i]});

    std::vector<TextureLoader::Options> options(paths.size());
    options[0].srgb_mode = SRGBMode::srgb;
    options[0].generate_mips = true;
    options[1].srgb_mode = SRGBMode::linear;
    options[1].max_mip_count = 2;
    options[2].max_mip_count = 1;

    TextureLoader loader{ref<Device>(ctx.device)};
    std::vector<ref<Texture>> expected;
    for (size_t i = 0; i < images.size(); ++i) {
        MemoryStream stream(bytes[i].data(), bytes[i].size());
        expected.push_back(loader.load_texture(&stream, options[i]));
    }

    for (uint64_t budget : {TextureLoader::DEFAULT_MEMORY_BUDGET, uint64_t(1)}) {
        TextureLoader bulk_loader{ref<Device>(ctx.device), budget};
        const auto actual = bulk_loader.load_textures(images, options);
        REQUIRE(actual.size() == expected.size());
        for (size_t i = 0; i < actual.size(); ++i)
            check_textures_match(actual[i], expected[i]);
    }

    TextureLoader::Options uniform_options{.srgb_mode = SRGBMode::linear};
    const std::array repeated_images{images[1], images[1]};
    const auto uniform_actual = loader.load_textures(repeated_images, uniform_options);
    REQUIRE(uniform_actual.size() == repeated_images.size());
    check_textures_match(uniform_actual[0], uniform_actual[1]);
}

TEST_CASE_GPU("texture_loader_encoded_image_error_name")
{
    const std::array<uint8_t, 4> invalid_bytes{0, 1, 2, 3};
    const std::array images{TextureLoader::EncodedImageView{
        .data = std::span<const uint8_t>(invalid_bytes),
        .name = "embedded://broken.png"
    }};
    TextureLoader loader{ref<Device>(ctx.device)};
    CHECK_THROWS_WITH_AS(loader.load_textures(images), doctest::Contains("embedded://broken.png"), std::runtime_error);
}

TEST_CASE_GPU("texture_loader_batched_uploads")
{
    constexpr size_t TEXTURE_COUNT = 33;
    ref<Bitmap> bitmap = ref(new Bitmap(Bitmap::PixelFormat::rgba, Bitmap::ComponentType::uint8, 8, 8));
    for (size_t i = 0; i < bitmap->buffer_size(); ++i)
        bitmap->uint8_data()[i] = static_cast<uint8_t>((i * 17 + 43) % 256);

    std::vector<const Bitmap*> bitmaps(TEXTURE_COUNT, bitmap.get());
    TextureLoader loader{ref<Device>(ctx.device)};
    TextureLoader::Options options{
        .srgb_mode = SRGBMode::linear,
        .generate_mips = true,
    };

    const std::vector<ref<Texture>> textures = loader.load_textures(bitmaps, options);
    REQUIRE(textures.size() == TEXTURE_COUNT);
    REQUIRE(textures.front()->mip_count() > 1);
    for (const ref<Texture>& texture : textures)
        check_textures_match(texture, textures.front());

    const ref<Texture> texture_array = loader.load_texture_array(bitmaps, options);
    REQUIRE(texture_array);
    REQUIRE(texture_array->array_length() == TEXTURE_COUNT);
    REQUIRE(texture_array->mip_count() > 1);
    for (uint32_t layer = 0; layer < texture_array->layer_count(); ++layer)
        check_subresources_match(texture_array, layer, texture_array, 0, 0);
}

TEST_SUITE_END();
