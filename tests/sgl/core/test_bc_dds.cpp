// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/bc_dds.h"
#include "sgl/core/memory_stream.h"
#include "sgl/core/platform.h"
#include "sgl/device/native_formats.h"

#include <cstring>

using namespace sgl;

TEST_SUITE_BEGIN("bc_dds");

TEST_CASE("bc_compressed_image_from_dds")
{
    std::filesystem::path images_dir = platform::project_directory() / "data" / "test_images" / "dds";

    SUBCASE("bc7_unorm_single_mip")
    {
        DDSFile dds(images_dir / "bc7-unorm-tiny.dds");
        BCCompressedImage image = bc_compressed_image_from_dds(dds);

        CHECK_EQ(image.format, BCFormat::bc7_unorm);
        CHECK_EQ(image.mip_levels.size(), 1);
        CHECK_EQ(image.mip_levels[0].width, 1);
        CHECK_EQ(image.mip_levels[0].height, 1);
        // 1x1 -> 1 block of 16 bytes for BC7
        CHECK_EQ(image.mip_levels[0].data.size(), 16);
    }

    SUBCASE("bc7_unorm_with_mipmaps")
    {
        DDSFile dds(images_dir / "bc7-unorm.dds");
        BCCompressedImage image = bc_compressed_image_from_dds(dds);

        CHECK_EQ(image.format, BCFormat::bc7_unorm);
        CHECK_EQ(image.mip_levels.size(), 9);
        CHECK_EQ(image.mip_levels[0].width, 256);
        CHECK_EQ(image.mip_levels[0].height, 256);
        CHECK_EQ(image.mip_levels[1].width, 128);
        CHECK_EQ(image.mip_levels[1].height, 128);
        CHECK_EQ(image.mip_levels[8].width, 1);
        CHECK_EQ(image.mip_levels[8].height, 1);
    }

    SUBCASE("bc1_unorm")
    {
        DDSFile dds(images_dir / "bc1-unorm.dds");
        BCCompressedImage image = bc_compressed_image_from_dds(dds);

        CHECK_EQ(image.format, BCFormat::bc1_unorm);
        CHECK_EQ(image.mip_levels.size(), 9);
        CHECK_EQ(image.mip_levels[0].width, 256);
        CHECK_EQ(image.mip_levels[0].height, 256);
        // BC1: 8 bytes per block, 64*64 = 4096 blocks for 256x256
        CHECK_EQ(image.mip_levels[0].data.size(), 4096 * 8);
    }

    SUBCASE("bc6h_uf16")
    {
        DDSFile dds(images_dir / "bc6h-uf16.dds");
        BCCompressedImage image = bc_compressed_image_from_dds(dds);

        CHECK_EQ(image.format, BCFormat::bc6h_ufloat);
        CHECK_EQ(image.mip_levels.size(), 9);
        CHECK_EQ(image.mip_levels[0].width, 409);
        CHECK_EQ(image.mip_levels[0].height, 204);
    }
}

TEST_CASE("bc_compressed_image_to_dds_roundtrip")
{
    std::filesystem::path images_dir = platform::project_directory() / "data" / "test_images" / "dds";

    SUBCASE("bc7_unorm_tiny")
    {
        DDSFile original(images_dir / "bc7-unorm-tiny.dds");
        BCCompressedImage image = bc_compressed_image_from_dds(original);

        // Write to memory stream.
        MemoryStream stream;
        bc_compressed_image_to_dds(image, &stream);

        // Read back as DDSFile.
        stream.seek(0);
        DDSFile read_back(&stream);

        CHECK_EQ(read_back.dxgi_format(), original.dxgi_format());
        CHECK_EQ(read_back.type(), original.type());
        CHECK_EQ(read_back.width(), original.width());
        CHECK_EQ(read_back.height(), original.height());
        CHECK_EQ(read_back.mip_count(), original.mip_count());
        CHECK_EQ(read_back.array_size(), original.array_size());

        // Verify resource data is byte-identical.
        CHECK_EQ(read_back.resource_size(), original.resource_size());
        CHECK(std::memcmp(read_back.resource_data(), original.resource_data(), original.resource_size()) == 0);
    }

    SUBCASE("bc7_unorm_with_mipmaps")
    {
        DDSFile original(images_dir / "bc7-unorm.dds");
        BCCompressedImage image = bc_compressed_image_from_dds(original);

        MemoryStream stream;
        bc_compressed_image_to_dds(image, &stream);

        stream.seek(0);
        DDSFile read_back(&stream);

        CHECK_EQ(read_back.dxgi_format(), original.dxgi_format());
        CHECK_EQ(read_back.width(), original.width());
        CHECK_EQ(read_back.height(), original.height());
        CHECK_EQ(read_back.mip_count(), original.mip_count());

        CHECK_EQ(read_back.resource_size(), original.resource_size());
        CHECK(std::memcmp(read_back.resource_data(), original.resource_data(), original.resource_size()) == 0);
    }

    SUBCASE("bc1_unorm")
    {
        DDSFile original(images_dir / "bc1-unorm.dds");
        BCCompressedImage image = bc_compressed_image_from_dds(original);

        MemoryStream stream;
        bc_compressed_image_to_dds(image, &stream);

        stream.seek(0);
        DDSFile read_back(&stream);

        CHECK_EQ(read_back.dxgi_format(), original.dxgi_format());
        CHECK_EQ(read_back.width(), original.width());
        CHECK_EQ(read_back.height(), original.height());
        CHECK_EQ(read_back.mip_count(), original.mip_count());

        CHECK_EQ(read_back.resource_size(), original.resource_size());
        CHECK(std::memcmp(read_back.resource_data(), original.resource_data(), original.resource_size()) == 0);
    }

    SUBCASE("bc6h_uf16")
    {
        DDSFile original(images_dir / "bc6h-uf16.dds");
        BCCompressedImage image = bc_compressed_image_from_dds(original);

        MemoryStream stream;
        bc_compressed_image_to_dds(image, &stream);

        stream.seek(0);
        DDSFile read_back(&stream);

        CHECK_EQ(read_back.dxgi_format(), original.dxgi_format());
        CHECK_EQ(read_back.width(), original.width());
        CHECK_EQ(read_back.height(), original.height());
        CHECK_EQ(read_back.mip_count(), original.mip_count());

        CHECK_EQ(read_back.resource_size(), original.resource_size());
        CHECK(std::memcmp(read_back.resource_data(), original.resource_data(), original.resource_size()) == 0);
    }
}

TEST_CASE("bc_compressed_image_to_dds_from_synthetic_data")
{
    // Create a synthetic BCCompressedImage and verify round-trip.
    BCCompressedImage image;
    image.format = BCFormat::bc7_unorm;

    // Single 4x4 block -> 16 bytes.
    BCCompressedMip mip;
    mip.width = 4;
    mip.height = 4;
    mip.data.resize(16);
    for (size_t i = 0; i < 16; ++i)
        mip.data[i] = static_cast<uint8_t>(i);
    image.mip_levels.push_back(std::move(mip));

    MemoryStream stream;
    bc_compressed_image_to_dds(image, &stream);

    stream.seek(0);
    DDSFile dds(&stream);

    CHECK_EQ(dds.width(), 4u);
    CHECK_EQ(dds.height(), 4u);
    CHECK_EQ(dds.mip_count(), 1u);
    CHECK_EQ(dds.array_size(), 1u);
    CHECK_EQ(dds.depth(), 1u);
    CHECK(dds.compressed());

    // Convert DXGI format back and verify it's BC7.
    Format format = get_format(DXGI_FORMAT(dds.dxgi_format()));
    auto bc_format = format_to_bc_format(format);
    REQUIRE(bc_format.has_value());
    CHECK_EQ(*bc_format, BCFormat::bc7_unorm);

    // Verify resource data.
    CHECK_EQ(dds.resource_size(), 16u);
    for (size_t i = 0; i < 16; ++i)
        CHECK_EQ(dds.resource_data()[i], static_cast<uint8_t>(i));

    // Full round-trip back to BCCompressedImage.
    BCCompressedImage rt = bc_compressed_image_from_dds(dds);
    CHECK_EQ(rt.format, BCFormat::bc7_unorm);
    CHECK_EQ(rt.mip_levels.size(), 1);
    CHECK_EQ(rt.mip_levels[0].width, 4u);
    CHECK_EQ(rt.mip_levels[0].height, 4u);
    CHECK_EQ(rt.mip_levels[0].data.size(), 16);
    CHECK(std::memcmp(rt.mip_levels[0].data.data(), image.mip_levels[0].data.data(), 16) == 0);
}

TEST_SUITE_END();
