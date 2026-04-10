// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/platform.h"

#include <cstdint>
#include <filesystem>

namespace sgl {

//
// NVTT3 types (copied from nvtt_wrapper.h - only the subset we use)
//

typedef enum {
    NVTT_False,
    NVTT_True,
} NvttBoolean;

typedef enum {
    NVTT_ValueType_UINT8,
    NVTT_ValueType_SINT8,
    NVTT_ValueType_FLOAT32,
} NvttValueType;

typedef enum {
    NVTT_ChannelOrder_Red = 0,
    NVTT_ChannelOrder_Green = 1,
    NVTT_ChannelOrder_Blue = 2,
    NVTT_ChannelOrder_Alpha = 3,
    NVTT_ChannelOrder_Zero = 4,
    NVTT_ChannelOrder_One = 5,
} NvttChannelOrder;

typedef struct {
    const void* data;
    int width;
    int height;
    int depth;
    int num_channels;
    NvttChannelOrder channel_swizzle[4];
    NvttBoolean channel_interleave;
} NvttRefImage;

typedef enum {
    NVTT_Format_BC1 = 1,
    NVTT_Format_BC2 = 3,
    NVTT_Format_BC3 = 4,
    NVTT_Format_BC4 = 7,
    NVTT_Format_BC4S = 8,
    NVTT_Format_BC5 = 10,
    NVTT_Format_BC5S = 11,
    NVTT_Format_BC6U = 14,
    NVTT_Format_BC6S = 15,
    NVTT_Format_BC7 = 16,
} NvttFormat;

typedef enum {
    NVTT_Quality_Fastest,
    NVTT_Quality_Normal,
    NVTT_Quality_Production,
    NVTT_Quality_Highest,
} NvttQuality;

typedef enum {
    NVTT_EncodeFlags_None = 0,
    NVTT_EncodeFlags_Opaque = 1 << 2,
} NvttEncodeFlags;

typedef struct {
    uint32_t sType;
    NvttFormat format;
    NvttQuality quality;
    int rgb_pixel_type;
    void* timing_context;
    uint32_t encode_flags;
} NvttEncodeSettings;

// Opaque types - only used as pointers.
struct NvttCPUInputBuffer;
struct NvttSurface;

typedef enum {
    NVTT_InputFormat_BGRA_8UB,
    NVTT_InputFormat_BGRA_8SB,
    NVTT_InputFormat_RGBA_16F,
    NVTT_InputFormat_RGBA_32F,
    NVTT_InputFormat_R_32F,
} NvttInputFormat;

typedef enum {
    NVTT_MipmapFilter_Box,
    NVTT_MipmapFilter_Triangle,
    NVTT_MipmapFilter_Kaiser,
    NVTT_MipmapFilter_Mitchell,
} NvttMipmapFilter;

//
// NvttAPI - dynamically-loaded NVTT3 function pointers
//

/// Dynamically-loaded NVTT3 API. Owns the library handle and function pointers.
struct NvttAPI {
    bool available = false;
    SharedLibraryHandle library = nullptr;

    // --- Encoding (low-level API) ---
    NvttCPUInputBuffer* (*nvttCreateCPUInputBuffer)(
        const NvttRefImage*,
        NvttValueType,
        int,
        int,
        int,
        float,
        float,
        float,
        float,
        void*,
        unsigned*
    ) = nullptr;
    void (*nvttDestroyCPUInputBuffer)(NvttCPUInputBuffer*) = nullptr;
    NvttBoolean (*nvttEncodeCPU)(const NvttCPUInputBuffer*, void*, const NvttEncodeSettings*) = nullptr;

    // --- Surface (for NVTT3 mipmap generation) ---
    NvttSurface* (*nvttCreateSurface)() = nullptr;
    void (*nvttDestroySurface)(NvttSurface*) = nullptr;
    NvttBoolean (*nvttSurfaceSetImageData)(
        NvttSurface*,
        NvttInputFormat,
        int,
        int,
        int,
        const void*,
        NvttBoolean,
        void*
    ) = nullptr;
    NvttBoolean (*nvttSurfaceBuildNextMipmapDefaults)(NvttSurface*, NvttMipmapFilter, int, void*) = nullptr;
    float* (*nvttSurfaceData)(NvttSurface*) = nullptr;
    int (*nvttSurfaceWidth)(const NvttSurface*) = nullptr;
    int (*nvttSurfaceHeight)(const NvttSurface*) = nullptr;

    // --- Version ---
    unsigned int (*nvttVersion)() = nullptr;

    ~NvttAPI()
    {
        if (library)
            platform::release_shared_library(library);
    }

    NvttAPI(const NvttAPI&) = delete;
    NvttAPI& operator=(const NvttAPI&) = delete;
    NvttAPI() = default;

    /// Try to load NVTT3 from known locations. Sets `available` on success.
    void try_load();

private:
    bool load_from_path(const std::filesystem::path& path);
    bool resolve_symbols();
};

/// Get the global NVTT3 API singleton (lazy-initialized).
NvttAPI& get_nvtt_api();

} // namespace sgl
