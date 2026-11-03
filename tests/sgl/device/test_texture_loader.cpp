// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/core/bitmap.h"
#include "sgl/core/dds_file.h"
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

void check_dds_texture(const Texture* texture, const DDSFile& dds, Format format, TextureType type)
{
    REQUIRE(texture->format() == format);
    REQUIRE(texture->type() == type);
    REQUIRE(texture->width() == dds.width());
    REQUIRE(texture->height() == dds.height());
    REQUIRE(texture->depth() == dds.depth());
    REQUIRE(texture->array_length() == dds.array_size());
    REQUIRE(texture->mip_count() == dds.mip_count());
    const uint32_t layers = dds.array_size() * (dds.type() == DDSFile::TextureType::texture_cube ? 6 : 1);
    REQUIRE(texture->layer_count() == layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        for (uint32_t mip = 0; mip < dds.mip_count(); ++mip) {
            const auto actual = texture->get_subresource_data(layer, mip);
            const auto layout = texture->get_subresource_layout(mip, 1);
            uint32_t row_pitch, slice_pitch;
            dds.get_subresource_pitch(mip, &row_pitch, &slice_pitch);
            REQUIRE(layout.row_pitch == row_pitch);
            REQUIRE(layout.slice_pitch == slice_pitch);
            for (uint32_t z = 0; z < layout.size.z; ++z) {
                for (size_t row = 0; row < layout.row_count; ++row) {
                    const auto* actual_row
                        = static_cast<const uint8_t*>(actual.data) + z * actual.slice_pitch + row * actual.row_pitch;
                    const auto* expected_row = dds.get_subresource_data(mip, layer) + z * slice_pitch + row * row_pitch;
                    CHECK(std::memcmp(actual_row, expected_row, row_pitch) == 0);
                }
            }
        }
    }
}

void check_dds_srgb_modes(
    Device* device,
    const std::filesystem::path& path,
    Format authored,
    Format linear,
    Format srgb,
    TextureType type = TextureType::texture_2d
)
{
    const auto data = read_file(path);
    DDSFile dds(path);
    TextureLoader loader{ref<Device>(device)};
    check_dds_texture(loader.load_texture(path), dds, authored, type);

    const std::vector<TextureLoader::Options> options{
        {.srgb_mode = SRGBMode::automatic},
        {.srgb_mode = SRGBMode::linear},
        {.srgb_mode = SRGBMode::srgb},
    };
    const Format expected[]{authored, linear, srgb};
    const std::vector<std::filesystem::path> paths(options.size(), path);
    const auto per_image_textures = loader.load_textures(paths, options);
    REQUIRE(per_image_textures.size() == options.size());
    for (size_t i = 0; i < options.size(); ++i) {
        CAPTURE(i);
        check_dds_texture(per_image_textures[i], dds, expected[i], type);
        check_dds_texture(loader.load_texture(path, options[i]), dds, expected[i], type);
        MemoryStream stream(data.data(), data.size());
        check_dds_texture(loader.load_texture(&stream, options[i]), dds, expected[i], type);
        const auto shared_options_textures = loader.load_textures(paths, options[i]);
        REQUIRE(shared_options_textures.size() == paths.size());
        for (const auto& texture : shared_options_textures)
            check_dds_texture(texture, dds, expected[i], type);
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

TEST_CASE_GPU("texture_loader_dds_srgb_modes")
{
    const auto directory = platform::project_directory() / "data" / "test_images" / "dds";
    struct TestCase {
        const char* filename;
        Format authored;
        Format linear;
        Format srgb;
    };
    const TestCase cases[]{
        // Legacy FourCC headers, including DXT1/3/5 and formats without sRGB variants.
        {"bc1-unorm.dds", Format::bc1_unorm, Format::bc1_unorm, Format::bc1_unorm_srgb},
        {"bc2-unorm.dds", Format::bc2_unorm, Format::bc2_unorm, Format::bc2_unorm_srgb},
        {"bc3-unorm.dds", Format::bc3_unorm, Format::bc3_unorm, Format::bc3_unorm_srgb},
        {"bc4-unorm.dds", Format::bc4_unorm, Format::bc4_unorm, Format::bc4_unorm},
        {"bc5-unorm.dds", Format::bc5_unorm, Format::bc5_unorm, Format::bc5_unorm},
        // DX10 headers, covering both authored color spaces and HDR without an sRGB variant.
        {"bc1-unorm-srgb.dds", Format::bc1_unorm_srgb, Format::bc1_unorm, Format::bc1_unorm_srgb},
        {"bc2-unorm-srgb.dds", Format::bc2_unorm_srgb, Format::bc2_unorm, Format::bc2_unorm_srgb},
        {"bc3-unorm-srgb-odd.dds", Format::bc3_unorm_srgb, Format::bc3_unorm, Format::bc3_unorm_srgb},
        {"bc7-unorm.dds", Format::bc7_unorm, Format::bc7_unorm, Format::bc7_unorm_srgb},
        {"bc7-unorm-srgb.dds", Format::bc7_unorm_srgb, Format::bc7_unorm, Format::bc7_unorm_srgb},
        {"bc6h-uf16.dds", Format::bc6h_ufloat, Format::bc6h_ufloat, Format::bc6h_ufloat},
    };
    for (const auto& item : cases) {
        CAPTURE(item.filename);
        check_dds_srgb_modes(ctx.device, directory / item.filename, item.authored, item.linear, item.srgb);
    }
}

TEST_CASE_GPU("texture_loader_dds_srgb_layout")
{
    // Small DX10 textures with distinct bytes in every mip, layer, face and depth slice.
    struct TestCase {
        DDSFile::TextureType dds_type;
        TextureType type;
        uint32_t depth;
        uint32_t array_size;
        size_t texel_count;
    };
    const TestCase cases[]{
        {DDSFile::TextureType::texture_2d, TextureType::texture_2d_array, 1, 2, (16 + 4 + 1) * 2},
        {DDSFile::TextureType::texture_cube, TextureType::texture_cube, 1, 1, (16 + 4 + 1) * 6},
        {DDSFile::TextureType::texture_3d, TextureType::texture_3d, 4, 1, 64 + 8 + 1},
    };
    const auto path = testing::get_case_temp_directory() / "layout.dds";
    for (const auto& item : cases) {
        CAPTURE(item.type);
        std::vector<uint8_t> data(item.texel_count * 4);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = static_cast<uint8_t>((i * 17 + 43) % 256);
        for (Format format : {Format::rgba8_unorm, Format::rgba8_unorm_srgb}) {
            DDSFile::write_dds(
                path,
                get_format_info(format).dxgi_format,
                item.dds_type,
                4,
                4,
                item.depth,
                3,
                item.array_size,
                data.data(),
                data.size()
            );
            check_dds_srgb_modes(ctx.device, path, format, Format::rgba8_unorm, Format::rgba8_unorm_srgb, item.type);
        }
    }
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
