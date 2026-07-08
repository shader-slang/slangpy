// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/dds_file.h"
#include "sgl/core/platform.h"
#include "sgl/core/memory_stream.h"
#include "sgl/device/native_formats.h"

#include <cstring>
#include <vector>

using namespace sgl;

TEST_SUITE_BEGIN("dds_file");

struct TestItem {
    const char* path;
    uint32_t dxgi_format;
    DDSFile::TextureType type;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t mip_count;
    uint32_t array_size;
    uint32_t row_pitch;
    uint32_t slice_pitch;
    uint32_t bits_per_pixel_or_block;
    uint32_t block_width;
    uint32_t block_height;
    bool compressed;
    bool srgb;
};

static const struct TestItem TEST_ITEMS[] = {
    {
        .path = "bc1-unorm.dds",
        .dxgi_format = DXGI_FORMAT_BC1_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 512,
        .slice_pitch = 32768,
        .bits_per_pixel_or_block = 64,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc1-unorm-srgb.dds",
        .dxgi_format = DXGI_FORMAT_BC1_UNORM_SRGB,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 512,
        .slice_pitch = 32768,
        .bits_per_pixel_or_block = 64,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = true,
    },
    {
        .path = "bc2-unorm.dds",
        .dxgi_format = DXGI_FORMAT_BC2_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 1024,
        .slice_pitch = 65536,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc2-unorm-srgb.dds",
        .dxgi_format = DXGI_FORMAT_BC2_UNORM_SRGB,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 1024,
        .slice_pitch = 65536,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = true,
    },
    {
        .path = "bc2-unorm-srgb-tiny.dds",
        .dxgi_format = DXGI_FORMAT_BC2_UNORM_SRGB,
        .type = DDSFile::TextureType::texture_2d,
        .width = 1,
        .height = 1,
        .depth = 1,
        .mip_count = 1,
        .array_size = 1,
        .row_pitch = 16,
        .slice_pitch = 16,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = true,
    },
    {
        .path = "bc3-unorm.dds",
        .dxgi_format = DXGI_FORMAT_BC3_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 1024,
        .slice_pitch = 65536,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc3-unorm-alpha.dds",
        .dxgi_format = DXGI_FORMAT_BC3_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 618,
        .height = 458,
        .depth = 1,
        .mip_count = 10,
        .array_size = 1,
        .row_pitch = 2480,
        .slice_pitch = 285200,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc3-unorm-alpha-tiny.dds",
        .dxgi_format = DXGI_FORMAT_BC3_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 3,
        .height = 3,
        .depth = 1,
        .mip_count = 2,
        .array_size = 1,
        .row_pitch = 16,
        .slice_pitch = 16,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc3-unorm-srgb.dds",
        .dxgi_format = DXGI_FORMAT_BC3_UNORM_SRGB,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 1024,
        .slice_pitch = 65536,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = true,
    },
    {
        .path = "bc3-unorm-srgb-odd.dds",
        .dxgi_format = DXGI_FORMAT_BC3_UNORM_SRGB,
        .type = DDSFile::TextureType::texture_2d,
        .width = 127,
        .height = 127,
        .depth = 1,
        .mip_count = 7,
        .array_size = 1,
        .row_pitch = 512,
        .slice_pitch = 16384,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = true,
    },
    {
        .path = "bc3-unorm-srgb-tiny.dds",
        .dxgi_format = DXGI_FORMAT_BC3_UNORM_SRGB,
        .type = DDSFile::TextureType::texture_2d,
        .width = 1,
        .height = 1,
        .depth = 1,
        .mip_count = 1,
        .array_size = 1,
        .row_pitch = 16,
        .slice_pitch = 16,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = true,
    },
    {
        .path = "bc4-unorm.dds",
        .dxgi_format = DXGI_FORMAT_BC4_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 512,
        .slice_pitch = 32768,
        .bits_per_pixel_or_block = 64,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc5-unorm.dds",
        .dxgi_format = DXGI_FORMAT_BC5_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 1024,
        .slice_pitch = 65536,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc5-unorm-tiny.dds",
        .dxgi_format = DXGI_FORMAT_BC5_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 1,
        .height = 1,
        .depth = 1,
        .mip_count = 1,
        .array_size = 1,
        .row_pitch = 16,
        .slice_pitch = 16,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc6h-uf16.dds",
        .dxgi_format = DXGI_FORMAT_BC6H_UF16,
        .type = DDSFile::TextureType::texture_2d,
        .width = 409,
        .height = 204,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 1648,
        .slice_pitch = 84048,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc7-unorm.dds",
        .dxgi_format = DXGI_FORMAT_BC7_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 1024,
        .slice_pitch = 65536,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc7-unorm-odd.dds",
        .dxgi_format = DXGI_FORMAT_BC7_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 127,
        .height = 127,
        .depth = 1,
        .mip_count = 7,
        .array_size = 1,
        .row_pitch = 512,
        .slice_pitch = 16384,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc7-unorm-srgb.dds",
        .dxgi_format = DXGI_FORMAT_BC7_UNORM_SRGB,
        .type = DDSFile::TextureType::texture_2d,
        .width = 256,
        .height = 256,
        .depth = 1,
        .mip_count = 9,
        .array_size = 1,
        .row_pitch = 1024,
        .slice_pitch = 65536,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = true,
    },
    {
        .path = "bc7-unorm-tiny.dds",
        .dxgi_format = DXGI_FORMAT_BC7_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 1,
        .height = 1,
        .depth = 1,
        .mip_count = 1,
        .array_size = 1,
        .row_pitch = 16,
        .slice_pitch = 16,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
};

static std::filesystem::path dds_test_path(const char* name)
{
    return platform::project_directory() / "data" / "test_images" / "dds" / name;
}

static std::vector<uint8_t> load_dds_data(const char* name)
{
    DDSFile file(dds_test_path(name));
    return std::vector<uint8_t>(file.data(), file.data() + file.size());
}

static void write_u32(std::vector<uint8_t>& data, size_t offset, uint32_t value)
{
    std::memcpy(data.data() + offset, &value, sizeof(value));
}

TEST_CASE("non_existing_file")
{
    CHECK_THROWS(DDSFile("__non_existing__.dds"));
}

TEST_CASE("invalid_header")
{
    std::filesystem::path path = platform::project_directory() / "data" / "test_images" / "dds" / "bc7-unorm-broken";
    CHECK_THROWS(DDSFile(path));
}

TEST_CASE("formats")
{
    std::filesystem::path images_dir = platform::project_directory() / "data" / "test_images" / "dds";

    for (const TestItem& item : TEST_ITEMS) {
        DDSFile file(images_dir / item.path);

#if 0
        printf("{\n");
        printf("    .path = \"%s\",\n", item.path);
        printf("    .dxgi_format = %d,\n", file.dxgi_format());
        printf("    .type = DDSFile::TextureType::%s,\n", enum_to_string(file.type()).c_str());
        printf("    .width = %d,\n", file.width());
        printf("    .height = %d,\n", file.height());
        printf("    .depth = %d,\n", file.depth());
        printf("    .mip_count = %d,\n", file.mip_count());
        printf("    .array_size = %d,\n", file.array_size());
        printf("    .row_pitch = %d,\n", file.row_pitch());
        printf("    .slice_pitch = %d,\n", file.slice_pitch());
        printf("    .bits_per_pixel_or_block = %d,\n", file.bits_per_pixel_or_block());
        printf("    .block_width = %d,\n", file.block_width());
        printf("    .block_height = %d,\n", file.block_height());
        printf("    .compressed = %s,\n", file.compressed() ? "true" : "false");
        printf("    .srgb = %s,\n", file.srgb() ? "true" : "false");
        printf("},\n");
#endif

        CHECK_EQ(file.dxgi_format(), item.dxgi_format);
        CHECK_EQ(file.type(), item.type);
        CHECK_EQ(file.width(), item.width);
        CHECK_EQ(file.height(), item.height);
        CHECK_EQ(file.depth(), item.depth);
        CHECK_EQ(file.mip_count(), item.mip_count);
        CHECK_EQ(file.array_size(), item.array_size);
        CHECK_EQ(file.row_pitch(), item.row_pitch);
        CHECK_EQ(file.slice_pitch(), item.slice_pitch);
        CHECK_EQ(file.bits_per_pixel_or_block(), item.bits_per_pixel_or_block);
        CHECK_EQ(file.block_width(), item.block_width);
        CHECK_EQ(file.block_height(), item.block_height);
        CHECK_EQ(file.compressed(), item.compressed);
        CHECK_EQ(file.srgb(), item.srgb);
    }
}

TEST_CASE("detect_dds_file")
{
    const uint32_t VALID_MAGIC = 0x20534444;
    MemoryStream valid_stream(&VALID_MAGIC, sizeof(VALID_MAGIC));
    CHECK(DDSFile::detect_dds_file(&valid_stream));

    const uint32_t INVALID_MAGIC = 0xffffffff;
    MemoryStream invalid_stream(&INVALID_MAGIC, sizeof(INVALID_MAGIC));
    CHECK_FALSE(DDSFile::detect_dds_file(&invalid_stream));
}

TEST_CASE("subresource_offsets_for_npot_mips")
{
    DDSFile file(dds_test_path("bc3-unorm-srgb-odd.dds"));

    size_t expected_offset = 0;
    for (uint32_t mip = 0; mip < file.mip_count(); ++mip) {
        CHECK_EQ(file.get_subresource_data(mip, 0), file.resource_data() + expected_offset);

        uint32_t slice_pitch;
        file.get_subresource_pitch(mip, nullptr, &slice_pitch);
        expected_offset += slice_pitch;
    }
    CHECK_LE(expected_offset, file.resource_size());
}

TEST_CASE("invalid_subresource_indices")
{
    DDSFile file(dds_test_path("bc7-unorm-tiny.dds"));

    CHECK_THROWS(file.get_subresource_data(file.mip_count(), 0));
    CHECK_THROWS(file.get_subresource_data(0, file.array_size()));
}

TEST_CASE("truncated_resource_data")
{
    std::vector<uint8_t> data = load_dds_data("bc7-unorm-tiny.dds");
    MemoryStream stream(static_cast<const void*>(data.data()), data.size() - 1);

    CHECK_THROWS(DDSFile(&stream));
}

TEST_CASE("3d_subresource_offsets")
{
    // Start from a valid DX10 BC7 header and turn it into a 4x4x4 volume with three mips.
    // BC7 uses one 16-byte block per 2D slice at every mip in this texture.
    std::vector<uint8_t> data = load_dds_data("bc7-unorm-tiny.dds");
    constexpr size_t header_size = 148;
    constexpr size_t resource_size = 112; // 16 * (4 + 2 + 1)
    data.resize(header_size + resource_size);

    write_u32(data, 12, 4);  // height
    write_u32(data, 16, 4);  // width
    write_u32(data, 24, 4);  // depth
    write_u32(data, 28, 3);  // mip count
    write_u32(data, 132, 4); // D3D11_RESOURCE_DIMENSION_TEXTURE3D
    write_u32(data, 140, 1); // array size

    MemoryStream stream(static_cast<const void*>(data.data()), data.size());
    DDSFile file(&stream);

    CHECK_EQ(file.type(), DDSFile::TextureType::texture_3d);
    CHECK_EQ(file.get_subresource_data(0, 0), file.resource_data());
    CHECK_EQ(file.get_subresource_data(0, 3), file.resource_data() + 48);
    CHECK_EQ(file.get_subresource_data(1, 0), file.resource_data() + 64);
    CHECK_EQ(file.get_subresource_data(1, 1), file.resource_data() + 80);
    CHECK_EQ(file.get_subresource_data(2, 0), file.resource_data() + 96);
    CHECK_THROWS(file.get_subresource_data(1, 2));
}

TEST_SUITE_END();
