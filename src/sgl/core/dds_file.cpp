// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "dds_file.h"

#include "sgl/core/file_stream.h"

// Adapted from https://github.com/redorav/ddspp

// Sources
// https://learn.microsoft.com/en-us/windows/uwp/gaming/complete-code-for-ddstextureloader
// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header-dxt10
// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-pixelformat

namespace sgl {

namespace detail {

    static constexpr uint32_t DDS_MAGIC = 0x20534444;

    static constexpr uint32_t DDS_ALPHAPIXELS = 0x00000001;
    static constexpr uint32_t DDS_ALPHA = 0x00000002;      // DDPF_ALPHA
    static constexpr uint32_t DDS_FOURCC = 0x00000004;     // DDPF_FOURCC
    static constexpr uint32_t DDS_RGB = 0x00000040;        // DDPF_RGB
    static constexpr uint32_t DDS_RGBA = 0x00000041;       // DDPF_RGB | DDPF_ALPHAPIXELS
    static constexpr uint32_t DDS_YUV = 0x00000200;        // DDPF_YUV
    static constexpr uint32_t DDS_LUMINANCE = 0x00020000;  // DDPF_LUMINANCE
    static constexpr uint32_t DDS_LUMINANCEA = 0x00020001; // DDPF_LUMINANCE | DDPF_ALPHAPIXELS

    static constexpr uint32_t DDS_PAL8 = 0x00000020;     // DDPF_PALETTEINDEXED8
    static constexpr uint32_t DDS_BUMPDUDV = 0x00080000; // DDPF_BUMPDUDV

    static constexpr uint32_t DDS_HEADER_FLAGS_CAPS = 0x00000001;        // DDSD_CAPS
    static constexpr uint32_t DDS_HEADER_FLAGS_HEIGHT = 0x00000002;      // DDSD_HEIGHT
    static constexpr uint32_t DDS_HEADER_FLAGS_WIDTH = 0x00000004;       // DDSD_WIDTH
    static constexpr uint32_t DDS_HEADER_FLAGS_PITCH = 0x00000008;       // DDSD_PITCH
    static constexpr uint32_t DDS_HEADER_FLAGS_PIXELFORMAT = 0x00001000; // DDSD_PIXELFORMAT
    static constexpr uint32_t DDS_HEADER_FLAGS_MIPMAP = 0x00020000;      // DDSD_MIPMAPCOUNT
    static constexpr uint32_t DDS_HEADER_FLAGS_LINEARSIZE = 0x00080000;  // DDSD_LINEARSIZE
    static constexpr uint32_t DDS_HEADER_FLAGS_VOLUME = 0x00800000;      // DDSD_DEPTH

    static constexpr uint32_t DDS_HEADER_CAPS_COMPLEX = 0x00000008; // DDSCAPS_COMPLEX
    static constexpr uint32_t DDS_HEADER_CAPS_MIPMAP = 0x00400000;  // DDSCAPS_MIPMAP
    static constexpr uint32_t DDS_HEADER_CAPS_TEXTURE = 0x00001000; // DDSCAPS_TEXTURE

    static constexpr uint32_t DDS_HEADER_CAPS2_CUBEMAP = 0x00000200;           // DDSCAPS2_CUBEMAP
    static constexpr uint32_t DDS_HEADER_CAPS2_CUBEMAP_POSITIVEX = 0x00000600; // DDSCAPS2_CUBEMAP |
                                                                               // DDSCAPS2_CUBEMAP_POSITIVEX
    static constexpr uint32_t DDS_HEADER_CAPS2_CUBEMAP_NEGATIVEX = 0x00000a00; // DDSCAPS2_CUBEMAP |
                                                                               // DDSCAPS2_CUBEMAP_NEGATIVEX
    static constexpr uint32_t DDS_HEADER_CAPS2_CUBEMAP_POSITIVEY = 0x00001200; // DDSCAPS2_CUBEMAP |
                                                                               // DDSCAPS2_CUBEMAP_POSITIVEY
    static constexpr uint32_t DDS_HEADER_CAPS2_CUBEMAP_NEGATIVEY = 0x00002200; // DDSCAPS2_CUBEMAP |
                                                                               // DDSCAPS2_CUBEMAP_NEGATIVEY
    static constexpr uint32_t DDS_HEADER_CAPS2_CUBEMAP_POSITIVEZ = 0x00004200; // DDSCAPS2_CUBEMAP |
                                                                               // DDSCAPS2_CUBEMAP_POSITIVEZ
    static constexpr uint32_t DDS_HEADER_CAPS2_CUBEMAP_NEGATIVEZ = 0x00008200; // DDSCAPS2_CUBEMAP |
                                                                               // DDSCAPS2_CUBEMAP_NEGATIVEZ
    static constexpr uint32_t DDS_HEADER_CAPS2_VOLUME = 0x00200000;            // DDSCAPS2_VOLUME
    static constexpr uint32_t DDS_HEADER_CAPS2_CUBEMAP_ALLFACES = DDS_HEADER_CAPS2_CUBEMAP_POSITIVEX
        | DDS_HEADER_CAPS2_CUBEMAP_NEGATIVEX | DDS_HEADER_CAPS2_CUBEMAP_POSITIVEY | DDS_HEADER_CAPS2_CUBEMAP_NEGATIVEY
        | DDS_HEADER_CAPS2_CUBEMAP_POSITIVEZ | DDS_HEADER_CAPS2_CUBEMAP_NEGATIVEZ;


    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_resource_misc_flag
    static constexpr uint32_t DXGI_MISC_FLAG_CUBEMAP = 0x2;
    static constexpr uint32_t DDS_MISC_FLAGS2_ALPHA_MODE_MASK = 0x7;

#define MAKE_FOURCC(a, b, c, d) ((a) + ((b) << 8) + ((c) << 16) + ((d) << 24))

    // FOURCC constants
    static constexpr uint32_t FOURCC_DXT1 = MAKE_FOURCC('D', 'X', 'T', '1');  // BC1_UNORM
    static constexpr uint32_t FOURCC_DXT2 = MAKE_FOURCC('D', 'X', 'T', '2');  // BC2_UNORM
    static constexpr uint32_t FOURCC_DXT3 = MAKE_FOURCC('D', 'X', 'T', '3');  // BC2_UNORM
    static constexpr uint32_t FOURCC_DXT4 = MAKE_FOURCC('D', 'X', 'T', '4');  // BC3_UNORM
    static constexpr uint32_t FOURCC_DXT5 = MAKE_FOURCC('D', 'X', 'T', '5');  // BC3_UNORM
    static constexpr uint32_t FOURCC_ATI1 = MAKE_FOURCC('A', 'T', 'I', '1');  // BC4_UNORM
    static constexpr uint32_t FOURCC_BC4U = MAKE_FOURCC('B', 'C', '4', 'U');  // BC4_UNORM
    static constexpr uint32_t FOURCC_BC4S = MAKE_FOURCC('B', 'C', '4', 'S');  // BC4_SNORM
    static constexpr uint32_t FOURCC_ATI2 = MAKE_FOURCC('A', 'T', 'I', '2');  // BC5_UNORM
    static constexpr uint32_t FOURCC_BC5U = MAKE_FOURCC('B', 'C', '5', 'U');  // BC5_UNORM
    static constexpr uint32_t FOURCC_BC5S = MAKE_FOURCC('B', 'C', '5', 'S');  // BC5_SNORM
    static constexpr uint32_t FOURCC_RGBG = MAKE_FOURCC('R', 'G', 'B', 'G');  // R8G8_B8G8_UNORM
    static constexpr uint32_t FOURCC_GRBG = MAKE_FOURCC('G', 'R', 'G', 'B');  // G8R8_G8B8_UNORM
    static constexpr uint32_t FOURCC_YUY2 = MAKE_FOURCC('Y', 'U', 'Y', '2');  // YUY2
    static constexpr uint32_t FOURCC_DXT10 = MAKE_FOURCC('D', 'X', '1', '0'); // DDS extension header

    // These values come from the original D3D9 D3DFORMAT values
    // https://learn.microsoft.com/en-us/windows/win32/direct3d9/d3dformat
    static constexpr uint32_t FOURCC_RGB8 = 20;
    static constexpr uint32_t FOURCC_A8R8G8B8 = 21;
    static constexpr uint32_t FOURCC_X8R8G8B8 = 22;
    static constexpr uint32_t FOURCC_R5G6B5 = 23; // B5G6R5_UNORM   (needs swizzling)
    static constexpr uint32_t FOURCC_X1R5G5B5 = 24;
    static constexpr uint32_t FOURCC_RGB5A1 = 25; // B5G5R5A1_UNORM (needs swizzling)
    static constexpr uint32_t FOURCC_RGBA4 = 26;  // B4G4R4A4_UNORM (needs swizzling)
    static constexpr uint32_t FOURCC_R3G3B2 = 27;
    static constexpr uint32_t FOURCC_A8 = 28;
    static constexpr uint32_t FOURCC_A8R3G3B2 = 29;
    static constexpr uint32_t FOURCC_X4R4G4B4 = 30;
    static constexpr uint32_t FOURCC_A2B10G10R10 = 31;
    static constexpr uint32_t FOURCC_A8B8G8R8 = 32;
    static constexpr uint32_t FOURCC_X8B8G8R8 = 33;
    static constexpr uint32_t FOURCC_G16R16 = 34;
    static constexpr uint32_t FOURCC_A2R10G10B10 = 35;
    static constexpr uint32_t FOURCC_RGBA16U = 36;  // R16G16B16A16_UNORM
    static constexpr uint32_t FOURCC_RGBA16S = 110; // R16G16B16A16_SNORM
    static constexpr uint32_t FOURCC_R16F = 111;    // R16_FLOAT
    static constexpr uint32_t FOURCC_RG16F = 112;   // R16G16_FLOAT
    static constexpr uint32_t FOURCC_RGBA16F = 113; // R16G16B16A16_FLOAT
    static constexpr uint32_t FOURCC_R32F = 114;    // R32_FLOAT
    static constexpr uint32_t FOURCC_RG32F = 115;   // R32G32_FLOAT
    static constexpr uint32_t FOURCC_RGBA32F = 116; // R32G32B32A32_FLOAT

    struct PixelFormat {
        uint32_t size;
        uint32_t flags;
        uint32_t fourCC;
        uint32_t RGBBitCount;
        uint32_t RBitMask;
        uint32_t GBitMask;
        uint32_t BBitMask;
        uint32_t ABitMask;
    };

    static_assert(sizeof(PixelFormat) == 32, "PixelFormat size mismatch");

    inline constexpr bool
    is_rgba_mask(const PixelFormat& ddspf, uint32_t rmask, uint32_t gmask, uint32_t bmask, uint32_t amask)
    {
        return (ddspf.RBitMask == rmask) && (ddspf.GBitMask == gmask) && (ddspf.BBitMask == bmask)
            && (ddspf.ABitMask == amask);
    }

    inline constexpr bool is_rgb_mask(const PixelFormat& ddspf, uint32_t rmask, uint32_t gmask, uint32_t bmask)
    {
        return (ddspf.RBitMask == rmask) && (ddspf.GBitMask == gmask) && (ddspf.BBitMask == bmask);
    }

    // https://docs.microsoft.com/en-us/windows/desktop/api/d3d11/ne-d3d11-d3d11_resource_dimension
    enum DXGIResourceDimension : unsigned char {
        DXGI_Unknown,
        DXGI_Buffer,
        DXGI_Texture1D,
        DXGI_Texture2D,
        DXGI_Texture3D
    };

    // Matches DXGI_FORMAT https://docs.microsoft.com/en-us/windows/desktop/api/dxgiformat/ne-dxgiformat-dxgi_format
    enum DXGIFormat : uint32_t {
        UNKNOWN = 0,
        R32G32B32A32_TYPELESS = 1,
        R32G32B32A32_FLOAT = 2,
        R32G32B32A32_UINT = 3,
        R32G32B32A32_SINT = 4,
        R32G32B32_TYPELESS = 5,
        R32G32B32_FLOAT = 6,
        R32G32B32_UINT = 7,
        R32G32B32_SINT = 8,
        R16G16B16A16_TYPELESS = 9,
        R16G16B16A16_FLOAT = 10,
        R16G16B16A16_UNORM = 11,
        R16G16B16A16_UINT = 12,
        R16G16B16A16_SNORM = 13,
        R16G16B16A16_SINT = 14,
        R32G32_TYPELESS = 15,
        R32G32_FLOAT = 16,
        R32G32_UINT = 17,
        R32G32_SINT = 18,
        R32G8X24_TYPELESS = 19,
        D32_FLOAT_S8X24_UINT = 20,
        R32_FLOAT_X8X24_TYPELESS = 21,
        X32_TYPELESS_G8X24_UINT = 22,
        R10G10B10A2_TYPELESS = 23,
        R10G10B10A2_UNORM = 24,
        R10G10B10A2_UINT = 25,
        R11G11B10_FLOAT = 26,
        R8G8B8A8_TYPELESS = 27,
        R8G8B8A8_UNORM = 28,
        R8G8B8A8_UNORM_SRGB = 29,
        R8G8B8A8_UINT = 30,
        R8G8B8A8_SNORM = 31,
        R8G8B8A8_SINT = 32,
        R16G16_TYPELESS = 33,
        R16G16_FLOAT = 34,
        R16G16_UNORM = 35,
        R16G16_UINT = 36,
        R16G16_SNORM = 37,
        R16G16_SINT = 38,
        R32_TYPELESS = 39,
        D32_FLOAT = 40,
        R32_FLOAT = 41,
        R32_UINT = 42,
        R32_SINT = 43,
        R24G8_TYPELESS = 44,
        D24_UNORM_S8_UINT = 45,
        R24_UNORM_X8_TYPELESS = 46,
        X24_TYPELESS_G8_UINT = 47,
        R8G8_TYPELESS = 48,
        R8G8_UNORM = 49,
        R8G8_UINT = 50,
        R8G8_SNORM = 51,
        R8G8_SINT = 52,
        R16_TYPELESS = 53,
        R16_FLOAT = 54,
        D16_UNORM = 55,
        R16_UNORM = 56,
        R16_UINT = 57,
        R16_SNORM = 58,
        R16_SINT = 59,
        R8_TYPELESS = 60,
        R8_UNORM = 61,
        R8_UINT = 62,
        R8_SNORM = 63,
        R8_SINT = 64,
        A8_UNORM = 65,
        R1_UNORM = 66,
        R9G9B9E5_SHAREDEXP = 67,
        R8G8_B8G8_UNORM = 68,
        G8R8_G8B8_UNORM = 69,
        BC1_TYPELESS = 70,
        BC1_UNORM = 71,
        BC1_UNORM_SRGB = 72,
        BC2_TYPELESS = 73,
        BC2_UNORM = 74,
        BC2_UNORM_SRGB = 75,
        BC3_TYPELESS = 76,
        BC3_UNORM = 77,
        BC3_UNORM_SRGB = 78,
        BC4_TYPELESS = 79,
        BC4_UNORM = 80,
        BC4_SNORM = 81,
        BC5_TYPELESS = 82,
        BC5_UNORM = 83,
        BC5_SNORM = 84,
        B5G6R5_UNORM = 85,
        B5G5R5A1_UNORM = 86,
        B8G8R8A8_UNORM = 87,
        B8G8R8X8_UNORM = 88,
        R10G10B10_XR_BIAS_A2_UNORM = 89,
        B8G8R8A8_TYPELESS = 90,
        B8G8R8A8_UNORM_SRGB = 91,
        B8G8R8X8_TYPELESS = 92,
        B8G8R8X8_UNORM_SRGB = 93,
        BC6H_TYPELESS = 94,
        BC6H_UF16 = 95,
        BC6H_SF16 = 96,
        BC7_TYPELESS = 97,
        BC7_UNORM = 98,
        BC7_UNORM_SRGB = 99,
        AYUV = 100,
        Y410 = 101,
        Y416 = 102,
        NV12 = 103,
        P010 = 104,
        P016 = 105,
        OPAQUE_420 = 106,
        YUY2 = 107,
        Y210 = 108,
        Y216 = 109,
        NV11 = 110,
        AI44 = 111,
        IA44 = 112,
        P8 = 113,
        A8P8 = 114,
        B4G4R4A4_UNORM = 115,

        P208 = 130,
        V208 = 131,
        V408 = 132,
        ASTC_4X4_TYPELESS = 133,
        ASTC_4X4_UNORM = 134,
        ASTC_4X4_UNORM_SRGB = 135,
        ASTC_5X4_TYPELESS = 137,
        ASTC_5X4_UNORM = 138,
        ASTC_5X4_UNORM_SRGB = 139,
        ASTC_5X5_TYPELESS = 141,
        ASTC_5X5_UNORM = 142,
        ASTC_5X5_UNORM_SRGB = 143,

        ASTC_6X5_TYPELESS = 145,
        ASTC_6X5_UNORM = 146,
        ASTC_6X5_UNORM_SRGB = 147,

        ASTC_6X6_TYPELESS = 149,
        ASTC_6X6_UNORM = 150,
        ASTC_6X6_UNORM_SRGB = 151,

        ASTC_8X5_TYPELESS = 153,
        ASTC_8X5_UNORM = 154,
        ASTC_8X5_UNORM_SRGB = 155,

        ASTC_8X6_TYPELESS = 157,
        ASTC_8X6_UNORM = 158,
        ASTC_8X6_UNORM_SRGB = 159,

        ASTC_8X8_TYPELESS = 161,
        ASTC_8X8_UNORM = 162,
        ASTC_8X8_UNORM_SRGB = 163,

        ASTC_10X5_TYPELESS = 165,
        ASTC_10X5_UNORM = 166,
        ASTC_10X5_UNORM_SRGB = 167,

        ASTC_10X6_TYPELESS = 169,
        ASTC_10X6_UNORM = 170,
        ASTC_10X6_UNORM_SRGB = 171,

        ASTC_10X8_TYPELESS = 173,
        ASTC_10X8_UNORM = 174,
        ASTC_10X8_UNORM_SRGB = 175,

        ASTC_10X10_TYPELESS = 177,
        ASTC_10X10_UNORM = 178,
        ASTC_10X10_UNORM_SRGB = 179,

        ASTC_12X10_TYPELESS = 181,
        ASTC_12X10_UNORM = 182,
        ASTC_12X10_UNORM_SRGB = 183,

        ASTC_12X12_TYPELESS = 185,
        ASTC_12X12_UNORM = 186,
        ASTC_12X12_UNORM_SRGB = 187,

        FORCE_UINT = 0xffffffff
    };

    struct Header {
        uint32_t size;
        uint32_t flags;
        uint32_t height;
        uint32_t width;
        uint32_t pitchOrLinearSize;
        uint32_t depth;
        uint32_t mipMapCount;
        uint32_t reserved1[11];
        PixelFormat ddspf;
        uint32_t caps;
        uint32_t caps2;
        uint32_t caps3;
        uint32_t caps4;
        uint32_t reserved2;
    };

    static_assert(sizeof(Header) == 124, "DDS Header size mismatch");

    struct HeaderDXT10 {
        DXGIFormat dxgiFormat;
        DXGIResourceDimension resourceDimension;
        uint32_t miscFlag;
        uint32_t arraySize;
        uint32_t reserved;
    };

    static_assert(sizeof(HeaderDXT10) == 20, "DDS DX10 Extended Header size mismatch");

    /// Minimum possible size of the header.
    static constexpr size_t MIN_HEADER_SIZE = sizeof(DDS_MAGIC) + sizeof(Header);

    inline constexpr bool is_dxt10(const Header& header)
    {
        return (header.ddspf.flags & DDS_FOURCC) && (header.ddspf.fourCC == FOURCC_DXT10);
    }

    inline constexpr bool is_compressed(DXGIFormat format)
    {
        return (format >= BC1_UNORM && format <= BC5_SNORM) || (format >= BC6H_TYPELESS && format <= BC7_UNORM_SRGB)
            || (format >= ASTC_4X4_TYPELESS && format <= ASTC_12X12_UNORM_SRGB);
    }

    inline constexpr bool is_srgb(DXGIFormat format)
    {
        switch (format) {
        case R8G8B8A8_UNORM_SRGB:
        case BC1_UNORM_SRGB:
        case BC2_UNORM_SRGB:
        case BC3_UNORM_SRGB:
        case B8G8R8A8_UNORM_SRGB:
        case B8G8R8X8_UNORM_SRGB:
        case BC7_UNORM_SRGB:
        case ASTC_4X4_UNORM_SRGB:
        case ASTC_5X4_UNORM_SRGB:
        case ASTC_5X5_UNORM_SRGB:
        case ASTC_6X5_UNORM_SRGB:
        case ASTC_6X6_UNORM_SRGB:
        case ASTC_8X5_UNORM_SRGB:
        case ASTC_8X6_UNORM_SRGB:
        case ASTC_8X8_UNORM_SRGB:
        case ASTC_10X5_UNORM_SRGB:
        case ASTC_10X6_UNORM_SRGB:
        case ASTC_10X8_UNORM_SRGB:
        case ASTC_10X10_UNORM_SRGB:
        case ASTC_12X10_UNORM_SRGB:
        case ASTC_12X12_UNORM_SRGB:
            return true;
        default:
            return false;
        }
    }

    inline constexpr uint32_t get_bits_per_pixel_or_block(DXGIFormat format)
    {
        if (format >= ASTC_4X4_TYPELESS && format <= ASTC_12X12_UNORM_SRGB)
            return 128; // All ASTC blocks are the same size

        switch (format) {
        case R1_UNORM:
            return 1;
        case R8_TYPELESS:
        case R8_UNORM:
        case R8_UINT:
        case R8_SNORM:
        case R8_SINT:
        case A8_UNORM:
        case AI44:
        case IA44:
        case P8:
            return 8;
        case NV12:
        case OPAQUE_420:
        case NV11:
            return 12;
        case R8G8_TYPELESS:
        case R8G8_UNORM:
        case R8G8_UINT:
        case R8G8_SNORM:
        case R8G8_SINT:
        case R16_TYPELESS:
        case R16_FLOAT:
        case D16_UNORM:
        case R16_UNORM:
        case R16_UINT:
        case R16_SNORM:
        case R16_SINT:
        case B5G6R5_UNORM:
        case B5G5R5A1_UNORM:
        case A8P8:
        case B4G4R4A4_UNORM:
            return 16;
        case P010:
        case P016:
            return 24;
        case BC1_UNORM:
        case BC1_UNORM_SRGB:
        case BC1_TYPELESS:
        case BC4_UNORM:
        case BC4_SNORM:
        case BC4_TYPELESS:
        case R16G16B16A16_TYPELESS:
        case R16G16B16A16_FLOAT:
        case R16G16B16A16_UNORM:
        case R16G16B16A16_UINT:
        case R16G16B16A16_SNORM:
        case R16G16B16A16_SINT:
        case R32G32_TYPELESS:
        case R32G32_FLOAT:
        case R32G32_UINT:
        case R32G32_SINT:
        case R32G8X24_TYPELESS:
        case D32_FLOAT_S8X24_UINT:
        case R32_FLOAT_X8X24_TYPELESS:
        case X32_TYPELESS_G8X24_UINT:
        case Y416:
        case Y210:
        case Y216:
            return 64;
        case R32G32B32_TYPELESS:
        case R32G32B32_FLOAT:
        case R32G32B32_UINT:
        case R32G32B32_SINT:
            return 96;
        case BC2_UNORM:
        case BC2_UNORM_SRGB:
        case BC2_TYPELESS:
        case BC3_UNORM:
        case BC3_UNORM_SRGB:
        case BC3_TYPELESS:
        case BC5_UNORM:
        case BC5_SNORM:
        case BC6H_UF16:
        case BC6H_SF16:
        case BC7_UNORM:
        case BC7_UNORM_SRGB:
        case R32G32B32A32_TYPELESS:
        case R32G32B32A32_FLOAT:
        case R32G32B32A32_UINT:
        case R32G32B32A32_SINT:
            return 128;
        default:
            return 32; // Most formats are 32 bits per pixel
            break;
        }
    }

    inline constexpr void get_block_size(DXGIFormat format, uint32_t& block_width, uint32_t& block_height)
    {
        switch (format) {
        case BC1_UNORM:
        case BC1_UNORM_SRGB:
        case BC1_TYPELESS:
        case BC4_UNORM:
        case BC4_SNORM:
        case BC4_TYPELESS:
        case BC2_UNORM:
        case BC2_UNORM_SRGB:
        case BC2_TYPELESS:
        case BC3_UNORM:
        case BC3_UNORM_SRGB:
        case BC3_TYPELESS:
        case BC5_UNORM:
        case BC5_SNORM:
        case BC6H_UF16:
        case BC6H_SF16:
        case BC7_UNORM:
        case BC7_UNORM_SRGB:
        case ASTC_4X4_TYPELESS:
        case ASTC_4X4_UNORM:
        case ASTC_4X4_UNORM_SRGB:
            block_width = 4;
            block_height = 4;
            break;
        case ASTC_5X4_TYPELESS:
        case ASTC_5X4_UNORM:
        case ASTC_5X4_UNORM_SRGB:
            block_width = 5;
            block_height = 4;
            break;
        case ASTC_5X5_TYPELESS:
        case ASTC_5X5_UNORM:
        case ASTC_5X5_UNORM_SRGB:
            block_width = 5;
            block_height = 5;
            break;
        case ASTC_6X5_TYPELESS:
        case ASTC_6X5_UNORM:
        case ASTC_6X5_UNORM_SRGB:
            block_width = 6;
            block_height = 5;
            break;
        case ASTC_6X6_TYPELESS:
        case ASTC_6X6_UNORM:
        case ASTC_6X6_UNORM_SRGB:
            block_width = 6;
            block_height = 6;
            break;
        case ASTC_8X5_TYPELESS:
        case ASTC_8X5_UNORM:
        case ASTC_8X5_UNORM_SRGB:
            block_width = 8;
            block_height = 5;
            break;
        case ASTC_8X6_TYPELESS:
        case ASTC_8X6_UNORM:
        case ASTC_8X6_UNORM_SRGB:
            block_width = 8;
            block_height = 6;
            break;
        case ASTC_8X8_TYPELESS:
        case ASTC_8X8_UNORM:
        case ASTC_8X8_UNORM_SRGB:
            block_width = 8;
            block_height = 8;
            break;
        case ASTC_10X5_TYPELESS:
        case ASTC_10X5_UNORM:
        case ASTC_10X5_UNORM_SRGB:
            block_width = 10;
            block_height = 5;
            break;
        case ASTC_10X6_TYPELESS:
        case ASTC_10X6_UNORM:
        case ASTC_10X6_UNORM_SRGB:
            block_width = 10;
            block_height = 6;
            break;
        case ASTC_10X8_TYPELESS:
        case ASTC_10X8_UNORM:
        case ASTC_10X8_UNORM_SRGB:
            block_width = 10;
            block_height = 8;
            break;
        case ASTC_10X10_TYPELESS:
        case ASTC_10X10_UNORM:
        case ASTC_10X10_UNORM_SRGB:
            block_width = 10;
            block_height = 10;
            break;
        case ASTC_12X10_TYPELESS:
        case ASTC_12X10_UNORM:
        case ASTC_12X10_UNORM_SRGB:
            block_width = 12;
            block_height = 10;
            break;
        case ASTC_12X12_TYPELESS:
        case ASTC_12X12_UNORM:
        case ASTC_12X12_UNORM_SRGB:
            block_width = 12;
            block_height = 12;
            break;
        default:
            block_width = 1;
            block_height = 1;
            break;
        }

        return;
    }

    inline constexpr uint32_t
    get_row_pitch(uint32_t width, uint32_t bits_per_pixel_or_block, uint32_t block_width, uint32_t mip)
    {
        // Shift width by mipmap index, round to next block size and round to next byte (for the rare less than 1 byte
        // per pixel formats) E.g. width = 119, mip = 3, BC1 compression
        // ((((119 >> 2) + 4 - 1) / 4) * 64) / 8 = 64 bytes
        return ((((width >> mip) + block_width - 1) / block_width) * bits_per_pixel_or_block + 7) / 8;
    }

} // namespace detail

using namespace detail;

DDSFile::DDSFile(Stream* stream)
{
    m_size = stream->size();
    if (m_size < MIN_HEADER_SIZE)
        SGL_THROW("DDS file is too small");

    m_data = new uint8_t[m_size];
    stream->read(m_data, m_size);

    if (!decode_header(m_data, m_size))
        SGL_THROW("DDS file has invalid header");
}

DDSFile::DDSFile(const std::filesystem::path& path)
    : DDSFile(ref(new FileStream(path, FileStream::Mode::read)))
{
}

DDSFile::~DDSFile()
{
    if (m_data)
        delete[] m_data;
}

const uint8_t* DDSFile::get_subresource_data(uint32_t mip, uint32_t slice) const
{
    size_t offset = 0;
    size_t mip0_size = m_slice_pitch * 8; // Work in bits

    if (m_type == TextureType::texture_3d) {
        for (uint32_t m = 0; m < mip; ++m) {
            size_t mip_size = mip0_size >> 2 * m;
            offset += mip_size * m_mip_count;
        }
        size_t last_mip = mip0_size >> 2 * mip;
        offset += last_mip * slice;
    } else {
        size_t mip_chain_size = 0;
        for (uint32_t m = 0; m < m_mip_count; ++m) {
            // Divide by 2 in width and height
            size_t mip_size = mip0_size >> 2 * m;
            mip_chain_size += mip_size > m_bits_per_pixel_or_block ? mip_size : m_bits_per_pixel_or_block;
        }
        offset += mip_chain_size * slice;
        for (uint32_t m = 0; m < mip; ++m) {
            // Divide by 2 in width and height
            size_t mip_size = mip0_size >> 2 * m;
            offset += mip_size > m_bits_per_pixel_or_block ? mip_size : m_bits_per_pixel_or_block;
        }
    }

    offset /= 8; // Back to bytes

    return m_data + m_header_size + offset;
}

void DDSFile::get_subresource_pitch(uint32_t mip, uint32_t* row_pitch, uint32_t* slice_pitch) const
{
    uint32_t row_pitch_ = get_row_pitch(m_width, m_bits_per_pixel_or_block, m_block_width, mip);
    uint32_t row_count = (std::max(1u, m_height >> mip) + m_block_height - 1) / m_block_height;
    uint32_t slice_pitch_ = row_count * row_pitch_;

    if (row_pitch)
        *row_pitch = row_pitch_;
    if (slice_pitch)
        *slice_pitch = slice_pitch_;
}

std::string DDSFile::to_string() const
{
    return fmt::format(
        "DDSFile(\n"
        "  dxgi_format = {},\n"
        "  type = {},\n"
        "  width = {},\n"
        "  height = {},\n"
        "  depth = {},\n"
        "  mip_count = {},\n"
        "  array_size = {}\n"
        ")",
        m_dxgi_format,
        m_type,
        m_width,
        m_height,
        m_depth,
        m_mip_count,
        m_array_size
    );
}

bool DDSFile::detect_dds_file(Stream* stream)
{
    size_t pos = stream->tell();
    uint32_t magic;
    stream->read(&magic, sizeof(magic));
    stream->seek(pos);
    return magic == DDS_MAGIC;
}

bool DDSFile::decode_header(const uint8_t* data, size_t size)
{
    // First 4 bytes are the magic DDS number
    const uint32_t magic = *reinterpret_cast<const uint32_t*>(data);
    if (magic != DDS_MAGIC)
        return false;

    const Header header = *reinterpret_cast<const Header*>(data + sizeof(DDS_MAGIC));
    const PixelFormat& ddspf = header.ddspf;
    bool dxt10 = is_dxt10(header);
    if (dxt10 && size < sizeof(DDS_MAGIC) + sizeof(Header) + sizeof(HeaderDXT10))
        return false;
    const HeaderDXT10 dxt10_header = *reinterpret_cast<const HeaderDXT10*>(data + sizeof(DDS_MAGIC) + sizeof(Header));

    // Read basic data from the header
    m_width = header.width > 0 ? header.width : 1;
    m_height = header.height > 0 ? header.height : 1;
    m_depth = header.depth > 0 ? header.depth : 1;
    m_mip_count = header.mipMapCount > 0 ? header.mipMapCount : 1;

    // Set some sensible defaults
    m_array_size = 1;
    m_srgb = false;
    m_type = TextureType::texture_2d;
    m_dxgi_format = UNKNOWN;

    if (dxt10) {
        m_dxgi_format = dxt10_header.dxgiFormat;

        m_array_size = dxt10_header.arraySize;

        switch (dxt10_header.resourceDimension) {
        case DXGI_Texture1D:
            m_depth = 1;
            m_type = TextureType::texture_1d;
            break;
        case DXGI_Texture2D:
            m_depth = 1;

            if (dxt10_header.miscFlag & DXGI_MISC_FLAG_CUBEMAP) {
                m_type = TextureType::texture_cube;
            } else {
                m_type = TextureType::texture_2d;
            }

            break;
        case DXGI_Texture3D:
            m_type = TextureType::texture_3d;
            m_array_size = 1; // There are no 3D texture arrays
            break;
        default:
            break;
        }
    } else {
        if (ddspf.flags & DDS_FOURCC) {
            uint32_t fourCC = ddspf.fourCC;

            switch (fourCC) {
            // Compressed
            case FOURCC_DXT1:
                m_dxgi_format = BC1_UNORM;
                break;
            case FOURCC_DXT2:
            case FOURCC_DXT3:
                m_dxgi_format = BC2_UNORM;
                break;
            case FOURCC_DXT4:
            case FOURCC_DXT5:
                m_dxgi_format = BC3_UNORM;
                break;
            case FOURCC_ATI1:
            case FOURCC_BC4U:
                m_dxgi_format = BC4_UNORM;
                break;
            case FOURCC_BC4S:
                m_dxgi_format = BC4_SNORM;
                break;
            case FOURCC_ATI2:
            case FOURCC_BC5U:
                m_dxgi_format = BC5_UNORM;
                break;
            case FOURCC_BC5S:
                m_dxgi_format = BC5_SNORM;
                break;

                // Video
            case FOURCC_RGBG:
                m_dxgi_format = R8G8_B8G8_UNORM;
                break;
            case FOURCC_GRBG:
                m_dxgi_format = G8R8_G8B8_UNORM;
                break;
            case FOURCC_YUY2:
                m_dxgi_format = YUY2;
                break;

                // Packed
            case FOURCC_R5G6B5:
                m_dxgi_format = B5G6R5_UNORM;
                break;
            case FOURCC_RGB5A1:
                m_dxgi_format = B5G5R5A1_UNORM;
                break;
            case FOURCC_RGBA4:
                m_dxgi_format = B4G4R4A4_UNORM;
                break;

                // Uncompressed
            case FOURCC_A8:
                m_dxgi_format = R8_UNORM;
                break;
            case FOURCC_A2B10G10R10:
                m_dxgi_format = R10G10B10A2_UNORM;
                break;
            case FOURCC_RGBA16U:
                m_dxgi_format = R16G16B16A16_UNORM;
                break;
            case FOURCC_RGBA16S:
                m_dxgi_format = R16G16B16A16_SNORM;
                break;
            case FOURCC_R16F:
                m_dxgi_format = R16_FLOAT;
                break;
            case FOURCC_RG16F:
                m_dxgi_format = R16G16_FLOAT;
                break;
            case FOURCC_RGBA16F:
                m_dxgi_format = R16G16B16A16_FLOAT;
                break;
            case FOURCC_R32F:
                m_dxgi_format = R32_FLOAT;
                break;
            case FOURCC_RG32F:
                m_dxgi_format = R32G32_FLOAT;
                break;
            case FOURCC_RGBA32F:
                m_dxgi_format = R32G32B32A32_FLOAT;
                break;
            default:
                break;
            }
        } else if (ddspf.flags & DDS_RGB) {
            switch (ddspf.RGBBitCount) {
            case 32:
                if (is_rgba_mask(ddspf, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000)) {
                    m_dxgi_format = R8G8B8A8_UNORM;
                } else if (is_rgba_mask(ddspf, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000)) {
                    m_dxgi_format = B8G8R8A8_UNORM;
                } else if (is_rgba_mask(ddspf, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000)) {
                    m_dxgi_format = B8G8R8X8_UNORM;
                }

                // No DXGI format maps to (0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000) aka D3DFMT_X8B8G8R8
                // No DXGI format maps to (0x000003ff, 0x000ffc00, 0x3ff00000, 0xc0000000) aka D3DFMT_A2R10G10B10

                // Note that many common DDS reader/writers (including D3DX) swap the
                // the RED/BLUE masks for 10:10:10:2 formats. We assume
                // below that the 'backwards' header mask is being used since it is most
                // likely written by D3DX. The more robust solution is to use the 'DX10'
                // header extension and specify the DXGI_FORMAT_R10G10B10A2_UNORM format directly

                // For 'correct' writers, this should be 0x000003ff, 0x000ffc00, 0x3ff00000 for RGB data
                else if (is_rgba_mask(ddspf, 0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000)) {
                    m_dxgi_format = R10G10B10A2_UNORM;
                } else if (is_rgba_mask(ddspf, 0x0000ffff, 0xffff0000, 0x00000000, 0x00000000)) {
                    m_dxgi_format = R16G16_UNORM;
                } else if (is_rgba_mask(ddspf, 0xffffffff, 0x00000000, 0x00000000, 0x00000000)) {
                    // The only 32-bit color channel format in D3D9 was R32F
                    m_dxgi_format = R32_FLOAT; // D3DX writes this out as a FourCC of 114
                }
                break;
            case 24:
                if (is_rgb_mask(ddspf, 0x00ff0000, 0x0000ff00, 0x000000ff)) {
                    m_dxgi_format = B8G8R8X8_UNORM;
                }
                break;
            case 16:
                if (is_rgba_mask(ddspf, 0x7c00, 0x03e0, 0x001f, 0x8000)) {
                    m_dxgi_format = B5G5R5A1_UNORM;
                } else if (is_rgba_mask(ddspf, 0xf800, 0x07e0, 0x001f, 0x0000)) {
                    m_dxgi_format = B5G6R5_UNORM;
                } else if (is_rgba_mask(ddspf, 0x0f00, 0x00f0, 0x000f, 0xf000)) {
                    m_dxgi_format = B4G4R4A4_UNORM;
                }
                // No DXGI format maps to (0x7c00, 0x03e0, 0x001f, 0x0000) aka D3DFMT_X1R5G5B5.
                // No DXGI format maps to (0x0f00, 0x00f0, 0x000f, 0x0000) aka D3DFMT_X4R4G4B4.
                // No 3:3:2, 3:3:2:8, or paletted DXGI formats aka D3DFMT_A8R3G3B2, D3DFMT_R3G3B2, D3DFMT_P8,
                // D3DFMT_A8P8, etc.
                break;
            default:
                break;
            }
        } else if (ddspf.flags & DDS_LUMINANCE) {
            switch (ddspf.RGBBitCount) {
            case 16:
                if (is_rgba_mask(ddspf, 0x0000ffff, 0x00000000, 0x00000000, 0x00000000)) {
                    m_dxgi_format = R16_UNORM; // D3DX10/11 writes this out as DX10 extension.
                }
                if (is_rgba_mask(ddspf, 0x000000ff, 0x00000000, 0x00000000, 0x0000ff00)) {
                    m_dxgi_format = R8G8_UNORM; // D3DX10/11 writes this out as DX10 extension.
                }
                break;
            case 8:
                if (is_rgba_mask(ddspf, 0x000000ff, 0x00000000, 0x00000000, 0x00000000)) {
                    m_dxgi_format = R8_UNORM; // D3DX10/11 writes this out as DX10 extension
                }

                // No DXGI format maps to (0x0f, 0x00, 0x00, 0xf0) aka D3DFMT_A4L4

                if (is_rgba_mask(ddspf, 0x000000ff, 0x00000000, 0x00000000, 0x0000ff00)) {
                    m_dxgi_format = R8G8_UNORM; // Some DDS writers assume the bitcount should be 8 instead of 16
                }
                break;
            }
        } else if (ddspf.flags & DDS_ALPHA) {
            if (ddspf.RGBBitCount == 8) {
                m_dxgi_format = A8_UNORM;
            }
        } else if (ddspf.flags & DDS_BUMPDUDV) {
            if (ddspf.RGBBitCount == 32) {
                if (is_rgba_mask(ddspf, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000)) {
                    m_dxgi_format = R8G8B8A8_SNORM; // D3DX10/11 writes this out as DX10 extension
                }
                if (is_rgba_mask(ddspf, 0x0000ffff, 0xffff0000, 0x00000000, 0x00000000)) {
                    m_dxgi_format = R16G16_SNORM; // D3DX10/11 writes this out as DX10 extension
                }

                // No DXGI format maps to (0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000) aka D3DFMT_A2W10V10U10
            } else if (ddspf.RGBBitCount == 16) {
                if (is_rgba_mask(ddspf, 0x000000ff, 0x0000ff00, 0x00000000, 0x00000000)) {
                    m_dxgi_format = R8G8_SNORM; // D3DX10/11 writes this out as DX10 extension
                }
            }
        }

        if ((header.flags & DDS_HEADER_FLAGS_VOLUME) || (header.caps2 & DDS_HEADER_CAPS2_VOLUME)) {
            m_type = TextureType::texture_3d;
        } else if (header.caps2 & DDS_HEADER_CAPS2_CUBEMAP) {
            if ((header.caps2 & DDS_HEADER_CAPS2_CUBEMAP_ALLFACES) != DDS_HEADER_CAPS2_CUBEMAP_ALLFACES) {
                return false;
            }

            m_type = TextureType::texture_cube;
            m_array_size = 1;
            m_depth = 1;
        } else {
            m_type = TextureType::texture_2d;
        }
    }

    m_compressed = is_compressed(DXGIFormat(m_dxgi_format));
    m_srgb = is_srgb(DXGIFormat(m_dxgi_format));
    m_bits_per_pixel_or_block = get_bits_per_pixel_or_block(DXGIFormat(m_dxgi_format));
    get_block_size(DXGIFormat(m_dxgi_format), m_block_width, m_block_height);

    get_subresource_pitch(0, &m_row_pitch, &m_slice_pitch);

    m_row_pitch = get_row_pitch(m_width, m_bits_per_pixel_or_block, m_block_width, 0);
    m_slice_pitch = m_row_pitch * m_height / m_block_height;

    m_header_size = sizeof(DDS_MAGIC) + sizeof(Header) + (dxt10 ? sizeof(HeaderDXT10) : 0);

    return true;
}

} // namespace sgl
