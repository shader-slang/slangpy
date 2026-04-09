# Phase 5: NVTT3 Dynamic Loading

Optional accelerated backend for all formats including BC6H encoding. Added after SW path is working end-to-end.

Reference: `nvtt_wrapper.h` in project root contains the full NVTT3 C API. Only a subset of functions are needed.

## 5a. NVTT3 Dynamic Loading

Follow the RenderDoc dynamic loading pattern from `src/sgl/utils/renderdoc.cpp`. The NVTT3 types, opaque forward declarations, and the `NvttAPI` struct live in a new header `src/sgl/core/nvtt_api.h`. The implementation (`NvttAPI::try_load()`, `resolve_symbols()`) lives in `bc_codec.cpp`. No NVTT3 header is shipped — the types we need are copied from `nvtt_wrapper.h`.

### Types to define in `nvtt_api.h`

Copied from `nvtt_wrapper.h` — only the subset we use:

```cpp
// From nvtt_wrapper.h — only the types we actually use.

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
    NVTT_Format_BC1 = 1,   // = NVTT_Format_DXT1
    NVTT_Format_BC2 = 3,   // = NVTT_Format_DXT3
    NVTT_Format_BC3 = 4,   // = NVTT_Format_DXT5
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
    uint32_t sType;  // Must be 1 (NVTT_EncodeSettings_Version_1)
    NvttFormat format;
    NvttQuality quality;
    int rgb_pixel_type;        // NvttPixelType — 0 = UnsignedNorm
    void* timing_context;      // NvttTimingContext* — always nullptr for us
    uint32_t encode_flags;     // Bitmask of NvttEncodeFlags
} NvttEncodeSettings;

// Opaque types — only used as pointers.
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
```

### Library loading

Library search order (same as plan):
1. `platform::runtime_directory()` — next to `sgl.dll` / `libsgl.so`
2. System `PATH` / `LD_LIBRARY_PATH` (just pass the bare DLL name to `platform::load_shared_library`)
3. `SGL_NVTT_PATH` environment variable

DLL names:
- Windows: `nvtt.dll`
- Linux: `libnvtt.so`
- macOS: `libnvtt.dylib`

### `NvttAPI` struct (in `nvtt_api.h`)

The NVTT3 dynamic loading state lives in its own struct in `src/sgl/core/nvtt_api.h`, separate from `BCCodec::Impl`. This keeps the DLL lifecycle and function pointers self-contained and reusable.

```cpp
/// Dynamically-loaded NVTT3 API. Owns the library handle and function pointers.
/// Declared in nvtt_api.h, implementation in bc_codec.cpp.
struct NvttAPI {
    bool available = false;
    platform::SharedLibraryHandle library = nullptr;

    // --- Encoding (low-level API) ---
    NvttCPUInputBuffer* (*nvttCreateCPUInputBuffer)(
        const NvttRefImage*, NvttValueType, int, int, int,
        float, float, float, float, void*, unsigned*) = nullptr;
    void (*nvttDestroyCPUInputBuffer)(NvttCPUInputBuffer*) = nullptr;
    NvttBoolean (*nvttEncodeCPU)(
        const NvttCPUInputBuffer*, void*, const NvttEncodeSettings*) = nullptr;

    // --- Surface (for NVTT3 mipmap generation) ---
    NvttSurface* (*nvttCreateSurface)() = nullptr;
    void (*nvttDestroySurface)(NvttSurface*) = nullptr;
    NvttBoolean (*nvttSurfaceSetImageData)(
        NvttSurface*, NvttInputFormat, int, int, int, const void*,
        NvttBoolean, void*) = nullptr;
    NvttBoolean (*nvttSurfaceBuildNextMipmapDefaults)(
        NvttSurface*, NvttMipmapFilter, int, void*) = nullptr;
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
```

Accessed via a global getter (defined in `bc_codec.cpp`) with `std::call_once`:

```cpp
static NvttAPI& get_nvtt_api()
{
    static NvttAPI api;
    static std::once_flag flag;
    std::call_once(flag, [&] { api.try_load(); });
    return api;
}
```

`BCCodec::Impl` then simply references the global:

```cpp
struct BCCodec::Impl {
    NvttAPI& nvtt = get_nvtt_api();
};
```

### `try_load()` logic

```
1. Try runtime_directory() / "nvtt.dll" (or platform equivalent)
2. If not found, try bare "nvtt.dll" (system PATH search)
3. If not found, check SGL_NVTT_PATH env var → try that path / "nvtt.dll"
4. If still not found, log_info("NVTT3 not found, BC6H encoding unavailable"), return
5. resolve_symbols() — get_proc_address for all function pointers
6. If any symbol missing, release library, set available = false, log_warn
7. Call nvttVersion() and log_info the version
8. Set available = true
```

Thread-safe via `std::call_once` in `get_nvtt_api()`. The DLL is loaded at most once per process and lives for the process lifetime.

### BCFormat → NvttFormat mapping

```cpp
static NvttFormat bc_format_to_nvtt(BCFormat f)
{
    switch (f) {
    case BCFormat::bc1_unorm:
    case BCFormat::bc1_unorm_srgb: return NVTT_Format_BC1;
    case BCFormat::bc2_unorm:
    case BCFormat::bc2_unorm_srgb: return NVTT_Format_BC2;
    case BCFormat::bc3_unorm:
    case BCFormat::bc3_unorm_srgb: return NVTT_Format_BC3;
    case BCFormat::bc4_unorm:      return NVTT_Format_BC4;
    case BCFormat::bc4_snorm:      return NVTT_Format_BC4S;
    case BCFormat::bc5_unorm:      return NVTT_Format_BC5;
    case BCFormat::bc5_snorm:      return NVTT_Format_BC5S;
    case BCFormat::bc6h_ufloat:    return NVTT_Format_BC6U;
    case BCFormat::bc6h_sfloat:    return NVTT_Format_BC6S;
    case BCFormat::bc7_unorm:
    case BCFormat::bc7_unorm_srgb: return NVTT_Format_BC7;
    default: return NVTT_Format_BC1;
    }
}
```

### BCEncodeQuality → NvttQuality mapping

```cpp
static NvttQuality quality_to_nvtt(BCEncodeQuality q)
{
    switch (q) {
    case BCEncodeQuality::fastest:    return NVTT_Quality_Fastest;
    case BCEncodeQuality::normal:     return NVTT_Quality_Normal;
    case BCEncodeQuality::production: return NVTT_Quality_Production;
    case BCEncodeQuality::highest:    return NVTT_Quality_Highest;
    default:                          return NVTT_Quality_Normal;
    }
}
```

## 5b. NVTT3 Encode Flow (single mip)

Encode a single `BCImage` → compressed block data using NVTT3's low-level API:

```
1. Build NvttRefImage from BCImage:
   - data = src.data
   - width/height = src.width/src.height, depth = 1
   - num_channels = src.channel_count (clamp to 4)
   - channel_swizzle = {Red, Green, Blue, Alpha} (identity)
   - channel_interleave = NVTT_True (our data is interleaved RGBA)

2. Determine NvttValueType:
   - uint8 component_type → NVTT_ValueType_UINT8
   - float32 → NVTT_ValueType_FLOAT32
   - Others: convert to float32 first (or uint8 for 8-bit types)

3. Create NvttCPUInputBuffer:
   nvttCreateCPUInputBuffer(&ref_image, value_type, 1 /*numImages*/,
       src.width, src.height, /*tile = full image*/
       weights[0], weights[1], weights[2], weights[3],
       nullptr /*tc*/, &num_tiles);

4. Build NvttEncodeSettings:
   - sType = 1 (NVTT_EncodeSettings_Version_1)
   - format = bc_format_to_nvtt(format)
   - quality = quality_to_nvtt(options.quality)
   - rgb_pixel_type = 0 (UnsignedNorm) or 4 (Float) for BC6H
   - timing_context = nullptr
   - encode_flags = has_alpha ? NVTT_EncodeFlags_None : NVTT_EncodeFlags_Opaque

5. Allocate output buffer: bc_compressed_size(width, height, format) bytes

6. Call nvttEncodeCPU(input_buffer, output.data(), &settings)

7. Destroy input buffer: nvttDestroyCPUInputBuffer(input_buffer)
```

## 5c. NVTT3 Mipmap Encode Flow

When `options.generate_mipmaps` is true and NVTT3 is available, use NVTT3's surface API for mipmap generation:

```
1. Encode mip level 0 using the single-mip flow above

2. Create NvttSurface, load source data:
   nvttCreateSurface() → surface
   nvttSurfaceSetImageData(surface, NVTT_InputFormat_RGBA_32F,
       src.width, src.height, 1, float_data, NVTT_False, nullptr)
   (Source must be converted to float32 RGBA first)

3. For each subsequent mip level:
   a. nvttSurfaceBuildNextMipmapDefaults(surface, NVTT_MipmapFilter_Box, 1, nullptr)
   b. Get mip dimensions: nvttSurfaceWidth(surface), nvttSurfaceHeight(surface)
   c. Get pixel data: nvttSurfaceData(surface) → float* (planar RGBARGBA...)
      NOTE: NVTT Surface stores data in planar layout (R plane, G plane, B plane, A plane),
      NOT interleaved. Must convert to interleaved before creating NvttRefImage for encoding.
   d. Convert planar → interleaved RGBA float32
   e. Build NvttRefImage from interleaved data
   f. Create NvttCPUInputBuffer & encode as in single-mip flow
   g. Destroy input buffer

4. Destroy surface: nvttDestroySurface(surface)
```

When NVTT3 is available and `prefer_nvtt` is true, use NVTT3's surface API for both mipmap generation and encoding. This is faster than the `Bitmap::resample` path and keeps the entire mip chain within NVTT3.

## 5d. Integration into `BCCodec::encode()`

Modify the existing `BCCodec::encode()` to choose backend per mip level:

When `prefer_nvtt` is true and NVTT3 is available, the entire encode (including mipmap generation) is delegated to a single `nvtt_encode()` helper that uses NVTT3's surface + low-level encode APIs. Otherwise, the existing SW path runs.

```cpp
auto& nvtt = m_impl->nvtt;
if (nvtt.available && options.prefer_nvtt) {
    return nvtt_encode(nvtt, src, format, options);  // handles mipmaps internally
}

// BC6H has no SW fallback.
if (is_bc6h(format))
    SGL_THROW("BCCodec::encode: BC6H encoding requires NVTT3 (not available)");

// ... existing SW mip generation + per-level encode switch ...
```

The `BCEncodeOptions::prefer_nvtt` field (already in `bc_types.h`, default true) controls whether NVTT3 is used when available. Setting it to false forces SW path (useful for testing or benchmarking).

## 5e. Changes to existing files

| File | Change |
|------|--------|
| `src/sgl/core/nvtt_api.h` | **New file.** NVTT3 types (enums, structs), opaque forward declarations, `NvttAPI` struct declaration |
| `src/sgl/core/bc_codec.cpp` | `#include "nvtt_api.h"`, `NvttAPI::try_load()` / `resolve_symbols()` implementation, `get_nvtt_api()`, `nvtt_encode()` helper, modify `encode()` to dispatch to NVTT3 |
| `src/sgl/core/bc_codec.h` | No changes needed (API already has `is_nvtt_available()`, `can_encode()`) |
| `src/sgl/core/bc_types.h` | No changes needed (`prefer_nvtt` already in `BCEncodeOptions`) |
| `src/sgl/CMakeLists.txt` | Add `nvtt_api.h` to header sources |
| `tests/sgl/core/test_bc_codec.cpp` | Add conditional NVTT3 test cases |

## 5f. NVTT3 Tests

Add to `tests/sgl/core/test_bc_codec.cpp`:

```cpp
TEST_CASE("bc_codec_nvtt3_encode" * doctest::skip(!BCCodec().is_nvtt_available()))
{
    // Test all formats through NVTT3 backend.
    // Same pattern as "bc_codec_roundtrip_64x64" but with prefer_nvtt=true forced.
    // Verify PSNR within expected bounds for each format.
}

TEST_CASE("bc_codec_nvtt3_bc6h" * doctest::skip(!BCCodec().is_nvtt_available()))
{
    // Test BC6H encode+decode roundtrip (float16 HDR data).
    // Generate synthetic HDR test image (float32 RGB with values > 1.0).
    // Encode as bc6h_ufloat and bc6h_sfloat.
    // Decode and verify reasonable PSNR for HDR content.
}

TEST_CASE("bc_codec_nvtt3_vs_sw" * doctest::skip(!BCCodec().is_nvtt_available()))
{
    // Encode same image with SW (prefer_nvtt=false) and NVTT3 (prefer_nvtt=true).
    // Both should decode to similar results (within format tolerance).
    // This validates that the NVTT3 path produces valid BC blocks.
}

TEST_CASE("bc_codec_nvtt3_mipmaps" * doctest::skip(!BCCodec().is_nvtt_available()))
{
    // Test mipmap generation + NVTT3 encoding for BC6H (the format that only NVTT3 supports).
    // Verify correct number of mip levels and that each level decodes correctly.
}
```

All NVTT3 tests use `doctest::skip()` so they are automatically skipped when NVTT3 is not available (CI without NVTT3 DLL still passes).

## Verification

1. Build: `cmake --build --preset windows-msvc-debug` — no new dependencies, no CMake changes needed
2. Without NVTT3: all existing tests still pass, NVTT3 tests skipped, `is_nvtt_available()` returns false
3. With NVTT3: place `nvtt.dll` next to `sgl.dll` in build output, NVTT3 tests run and pass
4. `pre-commit run --all-files` passes
