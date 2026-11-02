// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/memory_stream.h"

using namespace sgl;

TEST_SUITE_BEGIN("bitmap");

TEST_CASE("Bitmap_read_info")
{
    MemoryStream stream;
    const uint64_t prefix = 0;
    stream.write(&prefix, sizeof(prefix));
    Bitmap bitmap(Bitmap::PixelFormat::rgb, Bitmap::ComponentType::uint8, 2, 2);
    bitmap.clear();
    bitmap.write(&stream, Bitmap::FileFormat::bmp);
    // Keep only the BMP header, with no pixel payload.
    stream.truncate(sizeof(prefix) + 54);
    stream.seek(sizeof(prefix));
    const auto info = Bitmap::read_info(&stream);
    CHECK(info.width == 2);
    CHECK(info.height == 2);
    CHECK(info.channel_count == 3);
    CHECK(info.component_type == Bitmap::ComponentType::uint8);
    CHECK(stream.tell() == sizeof(prefix));
    stream.truncate(sizeof(prefix) + 2);
    CHECK_THROWS(Bitmap::read_info(&stream));
    CHECK(stream.tell() == sizeof(prefix));
}

TEST_SUITE_END();
