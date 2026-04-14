// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/dds_file.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/platform.h"
#include "sgl/core/memory_stream.h"
#include "sgl/device/native_formats.h"

#include <cstring>

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
        .slice_pitch = 4,
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
        .slice_pitch = 283960,
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
        .slice_pitch = 12,
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
        .slice_pitch = 16256,
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
        .slice_pitch = 4,
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
        .slice_pitch = 4,
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
        .slice_pitch = 16256,
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
        .slice_pitch = 4,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
};

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

TEST_CASE("write_read_roundtrip")
{
    std::filesystem::path images_dir = platform::project_directory() / "data" / "test_images" / "dds";

    for (const TestItem& item : TEST_ITEMS) {
        CAPTURE(item.path);

        DDSFile original(images_dir / item.path);

        // Write directly to memory stream using static write_dds.
        MemoryStream stream;
        DDSFile::write_dds(
            &stream,
            original.dxgi_format(),
            original.type(),
            original.width(),
            original.height(),
            original.depth(),
            original.mip_count(),
            original.array_size(),
            original.resource_data(),
            original.resource_size()
        );

        // Read back.
        stream.seek(0);
        DDSFile read_back(&stream);

        // Verify metadata matches.
        CHECK_EQ(read_back.dxgi_format(), original.dxgi_format());
        CHECK_EQ(read_back.type(), original.type());
        CHECK_EQ(read_back.width(), original.width());
        CHECK_EQ(read_back.height(), original.height());
        CHECK_EQ(read_back.depth(), original.depth());
        CHECK_EQ(read_back.mip_count(), original.mip_count());
        CHECK_EQ(read_back.array_size(), original.array_size());
        CHECK_EQ(read_back.row_pitch(), original.row_pitch());
        CHECK_EQ(read_back.slice_pitch(), original.slice_pitch());
        CHECK_EQ(read_back.bits_per_pixel_or_block(), original.bits_per_pixel_or_block());
        CHECK_EQ(read_back.block_width(), original.block_width());
        CHECK_EQ(read_back.block_height(), original.block_height());
        CHECK_EQ(read_back.compressed(), original.compressed());
        CHECK_EQ(read_back.srgb(), original.srgb());

        // Verify resource data is byte-identical.
        CHECK_EQ(read_back.resource_size(), original.resource_size());
        CHECK(std::memcmp(read_back.resource_data(), original.resource_data(), original.resource_size()) == 0);
    }
}

TEST_CASE("bitmap_read_dds")
{
    std::filesystem::path images_dir = platform::project_directory() / "data" / "test_images" / "dds";

    struct BitmapDDSTestItem {
        const char* path;
        uint32_t width;
        uint32_t height;
        Bitmap::PixelFormat pixel_format;
        Bitmap::ComponentType component_type;
        bool srgb;
    };

    static const BitmapDDSTestItem BITMAP_DDS_ITEMS[] = {
        {"bc1-unorm.dds", 256, 256, Bitmap::PixelFormat::rgba, Bitmap::ComponentType::uint8, false},
        {"bc1-unorm-srgb.dds", 256, 256, Bitmap::PixelFormat::rgba, Bitmap::ComponentType::uint8, true},
        {"bc3-unorm.dds", 256, 256, Bitmap::PixelFormat::rgba, Bitmap::ComponentType::uint8, false},
        {"bc3-unorm-srgb.dds", 256, 256, Bitmap::PixelFormat::rgba, Bitmap::ComponentType::uint8, true},
        {"bc4-unorm.dds", 256, 256, Bitmap::PixelFormat::r, Bitmap::ComponentType::uint8, false},
        {"bc5-unorm.dds", 256, 256, Bitmap::PixelFormat::rg, Bitmap::ComponentType::uint8, false},
        {"bc6h-uf16.dds", 409, 204, Bitmap::PixelFormat::rgb, Bitmap::ComponentType::float16, false},
        {"bc7-unorm.dds", 256, 256, Bitmap::PixelFormat::rgba, Bitmap::ComponentType::uint8, false},
        {"bc7-unorm-srgb.dds", 256, 256, Bitmap::PixelFormat::rgba, Bitmap::ComponentType::uint8, true},
    };

    for (const auto& item : BITMAP_DDS_ITEMS) {
        CAPTURE(item.path);

        try {
            Bitmap bmp(images_dir / item.path);

            CHECK_EQ(bmp.width(), item.width);
            CHECK_EQ(bmp.height(), item.height);
            CHECK_EQ(bmp.pixel_format(), item.pixel_format);
            CHECK_EQ(bmp.component_type(), item.component_type);
            CHECK_EQ(bmp.srgb_gamma(), item.srgb);
            CHECK_FALSE(bmp.empty());
            CHECK(bmp.buffer_size() > 0);
        } catch (const std::exception& e) {
            printf("EXCEPTION for %s: %s\n", item.path, e.what());
            CHECK(false);
        }
    }

    // Test auto-detection of DDS format.
    {
        Bitmap bmp(images_dir / "bc7-unorm.dds");
        CHECK_EQ(bmp.width(), 256);
        CHECK_EQ(bmp.height(), 256);
        CHECK_EQ(bmp.pixel_format(), Bitmap::PixelFormat::rgba);
    }

    // Test odd-sized DDS (non-multiple-of-4).
    {
        Bitmap bmp(images_dir / "bc7-unorm-odd.dds");
        CHECK_FALSE(bmp.empty());
        CHECK(bmp.buffer_size() > 0);
    }
}

TEST_SUITE_END();
