// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/dds_file.h"
#include "sgl/core/platform.h"
#include "sgl/core/memory_stream.h"
#include "sgl/device/native_formats.h"

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
    {
        .path = "bc7-unorm-nonsquare.dds",
        .dxgi_format = DXGI_FORMAT_BC7_UNORM,
        .type = DDSFile::TextureType::texture_2d,
        .width = 4,
        .height = 8,
        .depth = 1,
        .mip_count = 4,
        .array_size = 1,
        .row_pitch = 16,
        .slice_pitch = 32,
        .bits_per_pixel_or_block = 128,
        .block_width = 4,
        .block_height = 4,
        .compressed = true,
        .srgb = false,
    },
    {
        .path = "bc7-unorm-3d.dds",
        .dxgi_format = DXGI_FORMAT_BC7_UNORM,
        .type = DDSFile::TextureType::texture_3d,
        .width = 8,
        .height = 8,
        .depth = 4,
        .mip_count = 4,
        .array_size = 1,
        .row_pitch = 32,
        .slice_pitch = 64,
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

TEST_CASE("subresource_mip_clamp")
{
    std::filesystem::path path
        = platform::project_directory() / "data" / "test_images" / "dds" / "bc7-unorm-nonsquare.dds";
    DDSFile dds(path);

    {
        uint32_t row, slice;
        dds.get_subresource_pitch(0, &row, &slice);
        CHECK_EQ(row, 16);   // one 4-pixel block row
        CHECK_EQ(slice, 32); // two rows of blocks
    }
    {
        uint32_t row, slice;
        dds.get_subresource_pitch(1, &row, &slice);
        CHECK_EQ(row, 16);
        CHECK_EQ(slice, 16); // one row of one block
    }
    {
        uint32_t row, slice;
        dds.get_subresource_pitch(2, &row, &slice);
        CHECK_EQ(row, 16);
        CHECK_EQ(slice, 16);
    }
    // mip 3 hits width >> 3 == 0 - clamp ensures at least one block.
    {
        uint32_t row, slice;
        dds.get_subresource_pitch(3, &row, &slice);
        CHECK_EQ(row, 16);
        CHECK_EQ(slice, 16);
    }

    const uint8_t* base = dds.resource_data();
    CHECK_EQ(dds.get_subresource_data(0, 0) - base, 0);
    CHECK_EQ(dds.get_subresource_data(1, 0) - base, 32);
    CHECK_EQ(dds.get_subresource_data(2, 0) - base, 48);
    CHECK_EQ(dds.get_subresource_data(3, 0) - base, 64);

    // All subresource pointers are within bounds and monotonic.
    size_t total = dds.resource_size();
    CHECK_EQ(total, 80);
    const uint8_t* prev = nullptr;
    for (uint32_t m = 0; m < 4; ++m) {
        const uint8_t* ptr = dds.get_subresource_data(m, 0);
        size_t offset = ptr - base;
        CHECK(offset < total);
        if (prev)
            CHECK(ptr > prev);
        prev = ptr;
    }
}

TEST_CASE("subresource_3d")
{
    // 8x8x4 BC7 3D texture with 4 mips.
    std::filesystem::path path = platform::project_directory() / "data" / "test_images" / "dds" / "bc7-unorm-3d.dds";
    DDSFile dds(path);

    CHECK_EQ(dds.type(), DDSFile::TextureType::texture_3d);
    CHECK_EQ(dds.depth(), 4);
    CHECK_EQ(dds.mip_count(), 4);

    // Per-slice pitches.
    {
        uint32_t row, slice;
        dds.get_subresource_pitch(0, &row, &slice);
        CHECK_EQ(row, 32);
        CHECK_EQ(slice, 64);
    }
    {
        uint32_t row, slice;
        dds.get_subresource_pitch(1, &row, &slice);
        CHECK_EQ(row, 16);
        CHECK_EQ(slice, 16);
    }
    {
        uint32_t row, slice;
        dds.get_subresource_pitch(2, &row, &slice);
        CHECK_EQ(row, 16);
        CHECK_EQ(slice, 16);
    }

    const uint8_t* base = dds.resource_data();

    // 3D layout: each mip stores all depth slices contiguously.
    // mip 0 (depth=4): slice_pitch=64, 4x64=256 bytes
    CHECK_EQ(dds.get_subresource_data(0, 0) - base, 0);
    CHECK_EQ(dds.get_subresource_data(0, 1) - base, 64);
    CHECK_EQ(dds.get_subresource_data(0, 2) - base, 128);
    CHECK_EQ(dds.get_subresource_data(0, 3) - base, 192);

    // mip 1 (depth=2): slice_pitch=16, offset after mip 0 = 256
    CHECK_EQ(dds.get_subresource_data(1, 0) - base, 256);
    CHECK_EQ(dds.get_subresource_data(1, 1) - base, 272);

    // mip 2 (depth=1): slice_pitch=16, offset = 256 + 32 = 288
    CHECK_EQ(dds.get_subresource_data(2, 0) - base, 288);

    // mip 3 (depth=1): slice_pitch=16, offset = 288 + 16 = 304
    CHECK_EQ(dds.get_subresource_data(3, 0) - base, 304);

    // Subresource sizes match expected.
    {
        uint32_t row, slice;
        dds.get_subresource_pitch(0, &row, &slice);
        CHECK_EQ(size_t(slice) * std::max(1u, dds.depth() >> 0), 256);
    }
    {
        uint32_t row, slice;
        dds.get_subresource_pitch(1, &row, &slice);
        CHECK_EQ(size_t(slice) * std::max(1u, dds.depth() >> 1), 32);
    }
    {
        uint32_t row, slice;
        dds.get_subresource_pitch(2, &row, &slice);
        CHECK_EQ(size_t(slice) * std::max(1u, dds.depth() >> 2), 16);
    }
}

TEST_SUITE_END();
