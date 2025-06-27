// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "formats.h"

#include "sgl/device/native_formats.h"
#include "sgl/device/helpers.h"

#include "sgl/core/error.h"

namespace sgl {

static const FormatInfo s_format_infos[] = {
    // clang-format off
    // bpb = bytes_per_block
    // cc = channel_count
    // bw = block_width
    // bh = block_height
    // format                       name                        bpb     cc      type                        depth   stencil compressed  bw  bh      channel_bit_count   dxgi_format                             vk_format                           slang_format
    {Format::undefined,             "undefined",                0,      0,      FormatType::unknown,        false,  false,  false,      0,  0,      {0, 0, 0, 0    },   DXGI_FORMAT_UNKNOWN,                    VK_FORMAT_UNDEFINED,                nullptr},

    {Format::r8_uint,               "r8_uint",                  1,      1,      FormatType::uint,           false,  false,  false,      1,  1,      {8, 0, 0, 0    },   DXGI_FORMAT_R8_UINT,                    VK_FORMAT_R8_UINT,                  "r8ui"},
    {Format::r8_sint,               "r8_sint",                  1,      1,      FormatType::sint,           false,  false,  false,      1,  1,      {8, 0, 0, 0    },   DXGI_FORMAT_R8_SINT,                    VK_FORMAT_R8_SINT,                  "r8i"},
    {Format::r8_unorm,              "r8_unorm",                 1,      1,      FormatType::unorm,          false,  false,  false,      1,  1,      {8, 0, 0, 0    },   DXGI_FORMAT_R8_UNORM,                   VK_FORMAT_R8_UNORM,                 "r8"},
    {Format::r8_snorm,              "r8_snorm",                 1,      1,      FormatType::snorm,          false,  false,  false,      1,  1,      {8, 0, 0, 0    },   DXGI_FORMAT_R8_SNORM,                   VK_FORMAT_R8_SNORM,                 "r8_snorm"},

    {Format::rg8_uint,              "rg8_uint",                 2,      2,      FormatType::uint,           false,  false,  false,      1,  1,      {8, 8, 0, 0    },   DXGI_FORMAT_R8G8_UINT,                  VK_FORMAT_R8G8_UINT,                "rg8ui"},
    {Format::rg8_sint,              "rg8_sint",                 2,      2,      FormatType::sint,           false,  false,  false,      1,  1,      {8, 8, 0, 0    },   DXGI_FORMAT_R8G8_SINT,                  VK_FORMAT_R8G8_SINT,                "rg8i"},
    {Format::rg8_unorm,             "rg8_unorm",                2,      2,      FormatType::unorm,          false,  false,  false,      1,  1,      {8, 8, 0, 0    },   DXGI_FORMAT_R8G8_UNORM,                 VK_FORMAT_R8G8_UNORM,               "rg8"},
    {Format::rg8_snorm,             "rg8_snorm",                2,      2,      FormatType::snorm,          false,  false,  false,      1,  1,      {8, 8, 0, 0    },   DXGI_FORMAT_R8G8_SNORM,                 VK_FORMAT_R8G8_SNORM,               "rg8_snorm"},

    {Format::rgba8_uint,            "rgba8_uint",               4,      4,      FormatType::uint,           false,  false,  false,      1,  1,      {8, 8, 8, 8    },   DXGI_FORMAT_R8G8B8A8_UINT,              VK_FORMAT_R8G8B8A8_UINT,            "rgba8ui"},
    {Format::rgba8_sint,            "rgba8_sint",               4,      4,      FormatType::sint,           false,  false,  false,      1,  1,      {8, 8, 8, 8    },   DXGI_FORMAT_R8G8B8A8_SINT,              VK_FORMAT_R8G8B8A8_SINT,            "rgba8i"},
    {Format::rgba8_unorm,           "rgba8_unorm",              4,      4,      FormatType::unorm,          false,  false,  false,      1,  1,      {8, 8, 8, 8    },   DXGI_FORMAT_R8G8B8A8_UNORM,             VK_FORMAT_R8G8B8A8_UNORM,           "rgba8"},
    {Format::rgba8_unorm_srgb,      "rgba8_unorm_srgb",         4,      4,      FormatType::unorm_srgb,     false,  false,  false,      1,  1,      {8, 8, 8, 8    },   DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,        VK_FORMAT_R8G8B8A8_SRGB,            nullptr},
    {Format::rgba8_snorm,           "rgba8_snorm",              4,      4,      FormatType::snorm,          false,  false,  false,      1,  1,      {8, 8, 8, 8    },   DXGI_FORMAT_R8G8B8A8_SNORM,             VK_FORMAT_R8G8B8A8_SNORM,           "rgba8_snorm"},

    {Format::bgra8_unorm,           "bgra8_unorm",              4,      4,      FormatType::unorm,          false,  false,  false,      1,  1,      {8, 8, 8, 8    },   DXGI_FORMAT_B8G8R8A8_UNORM,             VK_FORMAT_B8G8R8A8_UNORM,           nullptr},
    {Format::bgra8_unorm_srgb,      "bgra8_unorm_srgb",         4,      4,      FormatType::unorm_srgb,     false,  false,  false,      1,  1,      {8, 8, 8, 8    },   DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,        VK_FORMAT_B8G8R8A8_SRGB,            nullptr},
    {Format::bgrx8_unorm,           "bgrx8_unorm",              4,      4,      FormatType::unorm,          false,  false,  false,      1,  1,      {8, 8, 8, 8    },   DXGI_FORMAT_B8G8R8X8_UNORM,             VK_FORMAT_UNDEFINED,                nullptr},
    {Format::bgrx8_unorm_srgb,      "bgrx8_unorm_srgb",         4,      4,      FormatType::unorm_srgb,     false,  false,  false,      1,  1,      {8, 8, 8, 8    },   DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,        VK_FORMAT_UNDEFINED,                nullptr},

    {Format::r16_uint,              "r16_uint",                 2,      1,      FormatType::uint,           false,  false,  false,      1,  1,      {16, 0, 0, 0   },   DXGI_FORMAT_R16_UINT,                   VK_FORMAT_R16_UINT,                 "r16ui"},
    {Format::r16_sint,              "r16_sint",                 2,      1,      FormatType::sint,           false,  false,  false,      1,  1,      {16, 0, 0, 0   },   DXGI_FORMAT_R16_SINT,                   VK_FORMAT_R16_SINT,                 "r16i"},
    {Format::r16_unorm,             "r16_unorm",                2,      1,      FormatType::unorm,          false,  false,  false,      1,  1,      {16, 0, 0, 0   },   DXGI_FORMAT_R16_UNORM,                  VK_FORMAT_R16_UNORM,                "r16"},
    {Format::r16_snorm,             "r16_snorm",                2,      1,      FormatType::snorm,          false,  false,  false,      1,  1,      {16, 0, 0, 0   },   DXGI_FORMAT_R16_SNORM,                  VK_FORMAT_R16_SNORM,                "r16_snorm"},
    {Format::r16_float,             "r16_float",                2,      1,      FormatType::float_,         false,  false,  false,      1,  1,      {16, 0, 0, 0   },   DXGI_FORMAT_R16_FLOAT,                  VK_FORMAT_R16_SFLOAT,               "r16f"},

    {Format::rg16_uint,             "rg16_uint",                4,      2,      FormatType::uint,           false,  false,  false,      1,  1,      {16, 16, 0, 0  },   DXGI_FORMAT_R16G16_UINT,                VK_FORMAT_R16G16_UINT,              "rg16ui"},
    {Format::rg16_sint,             "rg16_sint",                4,      2,      FormatType::sint,           false,  false,  false,      1,  1,      {16, 16, 0, 0  },   DXGI_FORMAT_R16G16_SINT,                VK_FORMAT_R16G16_SINT,              "rg16i"},
    {Format::rg16_unorm,            "rg16_unorm",               4,      2,      FormatType::unorm,          false,  false,  false,      1,  1,      {16, 16, 0, 0  },   DXGI_FORMAT_R16G16_UNORM,               VK_FORMAT_R16G16_UNORM,             "rg16"},
    {Format::rg16_snorm,            "rg16_snorm",               4,      2,      FormatType::snorm,          false,  false,  false,      1,  1,      {16, 16, 0, 0  },   DXGI_FORMAT_R16G16_SNORM,               VK_FORMAT_R16G16_SNORM,             "rg16_snorm"},
    {Format::rg16_float,            "rg16_float",               4,      2,      FormatType::float_,         false,  false,  false,      1,  1,      {16, 16, 0, 0  },   DXGI_FORMAT_R16G16_FLOAT,               VK_FORMAT_R16G16_SFLOAT,            "rg16f"},

    {Format::rgba16_uint,           "rgba16_uint",              8,      4,      FormatType::uint,           false,  false,  false,      1,  1,      {16, 16, 16, 16},   DXGI_FORMAT_R16G16B16A16_UINT,          VK_FORMAT_R16G16B16A16_UINT,        "rgba16ui"},
    {Format::rgba16_sint,           "rgba16_sint",              8,      4,      FormatType::sint,           false,  false,  false,      1,  1,      {16, 16, 16, 16},   DXGI_FORMAT_R16G16B16A16_SINT,          VK_FORMAT_R16G16B16A16_SINT,        "rgba16i"},
    {Format::rgba16_unorm,          "rgba16_unorm",             8,      4,      FormatType::unorm,          false,  false,  false,      1,  1,      {16, 16, 16, 16},   DXGI_FORMAT_R16G16B16A16_UNORM,         VK_FORMAT_R16G16B16A16_UNORM,       "rgba16"},
    {Format::rgba16_snorm,          "rgba16_snorm",             8,      4,      FormatType::snorm,          false,  false,  false,      1,  1,      {16, 16, 16, 16},   DXGI_FORMAT_R16G16B16A16_SNORM,         VK_FORMAT_R16G16B16A16_SNORM,       "rgba16_snorm"},
    {Format::rgba16_float,          "rgba16_float",             8,      4,      FormatType::float_,         false,  false,  false,      1,  1,      {16, 16, 16, 16},   DXGI_FORMAT_R16G16B16A16_FLOAT,         VK_FORMAT_R16G16B16A16_SFLOAT,      "rgba16f"},

    {Format::r32_uint,              "r32_uint",                 4,      1,      FormatType::uint,           false,  false,  false,      1,  1,      {32, 0, 0, 0   },   DXGI_FORMAT_R32_UINT,                   VK_FORMAT_R32_UINT,                 "r32ui"},
    {Format::r32_sint,              "r32_sint",                 4,      1,      FormatType::sint,           false,  false,  false,      1,  1,      {32, 0, 0, 0   },   DXGI_FORMAT_R32_SINT,                   VK_FORMAT_R32_SINT,                 "r32i"},
    {Format::r32_float,             "r32_float",                4,      1,      FormatType::float_,         false,  false,  false,      1,  1,      {32, 0, 0, 0   },   DXGI_FORMAT_R32_FLOAT,                  VK_FORMAT_R32_SFLOAT,               "r32f"},

    {Format::rg32_uint,             "rg32_uint",                8,      2,      FormatType::uint,           false,  false,  false,      1,  1,      {32, 32, 0, 0  },   DXGI_FORMAT_R32G32_UINT,                VK_FORMAT_R32G32_UINT,              "rg32ui"},
    {Format::rg32_sint,             "rg32_sint",                8,      2,      FormatType::sint,           false,  false,  false,      1,  1,      {32, 32, 0, 0  },   DXGI_FORMAT_R32G32_SINT,                VK_FORMAT_R32G32_SINT,              "rg32i"},
    {Format::rg32_float,            "rg32_float",               8,      2,      FormatType::float_,         false,  false,  false,      1,  1,      {32, 32, 0, 0  },   DXGI_FORMAT_R32G32_FLOAT,               VK_FORMAT_R32G32_SFLOAT,            "rg32f"},

    {Format::rgb32_uint,            "rgb32_uint",               12,     3,      FormatType::uint,           false,  false,  false,      1,  1,      {32, 32, 32, 0 },   DXGI_FORMAT_R32G32B32_UINT,             VK_FORMAT_R32G32B32_UINT,           nullptr},
    {Format::rgb32_sint,            "rgb32_sint",               12,     3,      FormatType::sint,           false,  false,  false,      1,  1,      {32, 32, 32, 0 },   DXGI_FORMAT_R32G32B32_SINT,             VK_FORMAT_R32G32B32_SINT,           nullptr},
    {Format::rgb32_float,           "rgb32_float",              12,     3,      FormatType::float_,         false,  false,  false,      1,  1,      {32, 32, 32, 0 },   DXGI_FORMAT_R32G32B32_FLOAT,            VK_FORMAT_R32G32B32_SFLOAT,         nullptr},

    {Format::rgba32_uint,           "rgba32_uint",              16,     4,      FormatType::uint,           false,  false,  false,      1,  1,      {32, 32, 32, 32},   DXGI_FORMAT_R32G32B32A32_UINT,          VK_FORMAT_R32G32B32A32_UINT,        "rgba32ui"},
    {Format::rgba32_sint,           "rgba32_sint",              16,     4,      FormatType::sint,           false,  false,  false,      1,  1,      {32, 32, 32, 32},   DXGI_FORMAT_R32G32B32A32_SINT,          VK_FORMAT_R32G32B32A32_SINT,        "rgba32i"},
    {Format::rgba32_float,          "rgba32_float",             16,     4,      FormatType::float_,         false,  false,  false,      1,  1,      {32, 32, 32, 32},   DXGI_FORMAT_R32G32B32A32_FLOAT,         VK_FORMAT_R32G32B32A32_SFLOAT,      "rgba32f"},

    {Format::r64_uint,              "r64_uint",                 8,      1,      FormatType::uint,           false,  false,  false,      1,  1,      {64, 0, 0, 0   },   DXGI_FORMAT_UNKNOWN,                    VK_FORMAT_R64_UINT,                 "r64ui"},
    {Format::r64_sint,              "r64_sint",                 8,      1,      FormatType::sint,           false,  false,  false,      1,  1,      {64, 0, 0, 0   },   DXGI_FORMAT_UNKNOWN,                    VK_FORMAT_R64_SINT,                 "r64i"},

    {Format::bgra4_unorm,           "bgra4_unorm",              2,      4,      FormatType::unorm,          false,  false,  false,      1,  1,      {4, 4, 4, 4    },   DXGI_FORMAT_B4G4R4A4_UNORM,             VK_FORMAT_B4G4R4A4_UNORM_PACK16,    nullptr},
    {Format::b5g6r5_unorm,          "b5g6r5_unorm",             2,      3,      FormatType::unorm,          false,  false,  false,      1,  1,      {5, 6, 5, 0    },   DXGI_FORMAT_B5G6R5_UNORM,               VK_FORMAT_B5G6R5_UNORM_PACK16,      nullptr},
    {Format::bgr5a1_unorm,          "bgr5a1_unorm",             2,      4,      FormatType::unorm,          false,  false,  false,      1,  1,      {5, 5, 5, 1    },   DXGI_FORMAT_B5G5R5A1_UNORM,             VK_FORMAT_B5G5R5A1_UNORM_PACK16,    nullptr},

    {Format::rgb9e5_ufloat,         "rgb9e5_ufloat",            4,      3,      FormatType::float_,         false,  false,  false,      1,  1,      {9, 9, 9, 5    },   DXGI_FORMAT_R9G9B9E5_SHAREDEXP,         VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,   nullptr},
    {Format::rgb10a2_uint,          "rgb10a2_uint",             4,      4,      FormatType::uint,           false,  false,  false,      1,  1,      {10, 10, 10, 2 },   DXGI_FORMAT_R10G10B10A2_UINT,           VK_FORMAT_A2R10G10B10_UINT_PACK32,  "rgb10_a2ui"},
    {Format::rgb10a2_unorm,         "rgb10a2_unorm",            4,      4,      FormatType::unorm,          false,  false,  false,      1,  1,      {10, 10, 10, 2 },   DXGI_FORMAT_R10G10B10A2_UNORM,          VK_FORMAT_A2B10G10R10_UNORM_PACK32, "rgb10_a2"},
    {Format::r11g11b10_float,       "r11g11b10_float",          4,      3,      FormatType::float_,         false,  false,  false,      1,  1,      {11, 11, 10, 0 },   DXGI_FORMAT_R11G11B10_FLOAT,            VK_FORMAT_B10G11R11_UFLOAT_PACK32,  "r11f_g11f_b10f"},

    {Format::d32_float,             "d32_float",                4,      1,      FormatType::float_,         true,   false,  false,      1,  1,      {32, 0, 0, 0   },   DXGI_FORMAT_D32_FLOAT,                  VK_FORMAT_D32_SFLOAT,               nullptr},
    {Format::d16_unorm,             "d16_unorm",                2,      1,      FormatType::unorm,          true,   false,  false,      1,  1,      {16, 0, 0, 0   },   DXGI_FORMAT_D16_UNORM,                  VK_FORMAT_D16_UNORM,                nullptr},
    {Format::d32_float_s8_uint,     "d32_float_s8_uint",        8,      2,      FormatType::float_,         true,   true,   false,      1,  1,      {32, 8, 0, 0   },   DXGI_FORMAT_D32_FLOAT_S8X24_UINT,       VK_FORMAT_D32_SFLOAT_S8_UINT,       nullptr},

    {Format::bc1_unorm,             "bc1_unorm",                8,      4,      FormatType::unorm,          false,  false,  true,       4,  4,      {64, 0, 0, 0   },   DXGI_FORMAT_BC1_UNORM,                  VK_FORMAT_BC1_RGB_UNORM_BLOCK,      nullptr},
    {Format::bc1_unorm_srgb,        "bc1_unorm_srgb",           8,      4,      FormatType::unorm_srgb,     false,  false,  true,       4,  4,      {64, 0, 0, 0   },   DXGI_FORMAT_BC1_UNORM_SRGB,             VK_FORMAT_BC1_RGB_SRGB_BLOCK,       nullptr},
    {Format::bc2_unorm,             "bc2_unorm",                16,     4,      FormatType::unorm,          false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC2_UNORM,                  VK_FORMAT_BC2_UNORM_BLOCK,          nullptr},
    {Format::bc2_unorm_srgb,        "bc2_unorm_srgb",           16,     4,      FormatType::unorm_srgb,     false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC2_UNORM_SRGB,             VK_FORMAT_BC2_SRGB_BLOCK,           nullptr},
    {Format::bc3_unorm,             "bc3_unorm",                16,     4,      FormatType::unorm,          false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC3_UNORM,                  VK_FORMAT_BC3_UNORM_BLOCK,          nullptr},
    {Format::bc3_unorm_srgb,        "bc3_unorm_srgb",           16,     4,      FormatType::unorm_srgb,     false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC3_UNORM_SRGB,             VK_FORMAT_BC3_SRGB_BLOCK,           nullptr},
    {Format::bc4_unorm,             "bc4_unorm",                8,      1,      FormatType::unorm,          false,  false,  true,       4,  4,      {64, 0, 0, 0   },   DXGI_FORMAT_BC4_UNORM,                  VK_FORMAT_BC4_UNORM_BLOCK,          nullptr},
    {Format::bc4_snorm,             "bc4_snorm",                8,      1,      FormatType::snorm,          false,  false,  true,       4,  4,      {64, 0, 0, 0   },   DXGI_FORMAT_BC4_SNORM,                  VK_FORMAT_BC4_SNORM_BLOCK,          nullptr},
    {Format::bc5_unorm,             "bc5_unorm",                16,     2,      FormatType::unorm,          false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC5_UNORM,                  VK_FORMAT_BC5_UNORM_BLOCK,          nullptr},
    {Format::bc5_snorm,             "bc5_snorm",                16,     2,      FormatType::snorm,          false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC5_SNORM,                  VK_FORMAT_BC5_SNORM_BLOCK,          nullptr},
    {Format::bc6h_ufloat,           "bc6h_ufloat",              16,     3,      FormatType::float_,         false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC6H_UF16,                  VK_FORMAT_BC6H_UFLOAT_BLOCK,        nullptr},
    {Format::bc6h_sfloat,           "bc6h_sfloat",              16,     3,      FormatType::float_,         false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC6H_SF16,                  VK_FORMAT_BC6H_SFLOAT_BLOCK,        nullptr},
    {Format::bc7_unorm,             "bc7_unorm",                16,     4,      FormatType::unorm,          false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC7_UNORM,                  VK_FORMAT_BC7_UNORM_BLOCK,          nullptr},
    {Format::bc7_unorm_srgb,        "bc7_unorm_srgb",           16,     4,      FormatType::unorm_srgb,     false,  false,  true,       4,  4,      {128, 0, 0, 0  },   DXGI_FORMAT_BC7_UNORM_SRGB,             VK_FORMAT_BC7_SRGB_BLOCK,           nullptr},
    // clang-format on
};

static_assert(std::size(s_format_infos) == uint32_t(Format::count), "Format info table size mismatch");

std::string FormatInfo::to_string() const
{
    return fmt::format(
        "FormatInfo(\n"
        "  format = {},\n"
        "  bytes_per_block = {},\n"
        "  channel_count = {},\n"
        "  type = {},\n"
        "  is_depth = {},\n"
        "  is_stencil = {},\n"
        "  is_compressed = {},\n"
        "  block_width = {},\n"
        "  block_height = {},\n"
        "  channel_bit_count = [{}, {}, {}, {}],\n"
        "  dxgi_format = {},\n"
        "  vk_format = {}\n"
        ")",
        format,
        bytes_per_block,
        channel_count,
        type,
        is_depth,
        is_stencil,
        is_compressed,
        block_width,
        block_height,
        channel_bit_count[0],
        channel_bit_count[1],
        channel_bit_count[2],
        channel_bit_count[3],
        dxgi_format,
        vk_format
    );
}

const FormatInfo& get_format_info(Format format)
{
    SGL_ASSERT(uint32_t(format) < uint32_t(Format::count));
    return s_format_infos[uint32_t(format)];
}

DXGI_FORMAT get_dxgi_format(Format format)
{
    SGL_ASSERT(s_format_infos[uint32_t(format)].format == format);
    return DXGI_FORMAT(s_format_infos[uint32_t(format)].dxgi_format);
}

Format get_format(DXGI_FORMAT dxgi_format)
{
    for (uint32_t i = 0; i < uint32_t(Format::count); ++i)
        if (DXGI_FORMAT(s_format_infos[i].dxgi_format) == dxgi_format)
            return s_format_infos[i].format;
    return Format::undefined;
}

VkFormat get_vk_format(Format format)
{
    SGL_ASSERT(s_format_infos[uint32_t(format)].format == format);
    return VkFormat(s_format_infos[uint32_t(format)].vk_format);
}

Format get_format(VkFormat vk_format)
{
    for (uint32_t i = 0; i < uint32_t(Format::count); ++i)
        if (VkFormat(s_format_infos[i].vk_format) == vk_format)
            return s_format_infos[i].format;
    return Format::undefined;
}

} // namespace sgl
