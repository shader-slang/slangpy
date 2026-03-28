# Phase 3: BC Codec Implementation

Depends on Phase 1 (external deps) and Phase 2 (Bitmap mipmap).

## 3a. Image Types

Define lightweight non-owning views into CPU pixel data in `src/sgl/core/bc_types.h`:

```
using BCComponentType = DataStruct::Type; // uint8, float16, float32, etc.

struct BCImage {
    const void* data;
    uint32_t width, height;
    uint32_t row_pitch;          // bytes per row (allows stride)
    uint32_t channel_count;      // 1-4
    BCComponentType component_type; // uint8, float16, float32, etc.
};

struct BCMutableImage {
    void* data;
    uint32_t width, height;
    uint32_t row_pitch;
    uint32_t channel_count;
    BCComponentType component_type;
};
```

No `Bitmap` overloads on `BCCodec` — callers construct `BCImage`/`BCMutableImage` themselves. A free helper `bc_image_from_bitmap(const Bitmap&) -> BCImage` is provided for convenience.

## 3b. Encode/Decode Options & Result Types

All types in this section go into `src/sgl/core/bc_types.h`.

```
enum class BCFormat {
    bc1_unorm, bc1_unorm_srgb,
    bc2_unorm, bc2_unorm_srgb,
    bc3_unorm, bc3_unorm_srgb,
    bc4_unorm, bc4_snorm,
    bc5_unorm, bc5_snorm,
    bc6h_ufloat, bc6h_sfloat,
    bc7_unorm, bc7_unorm_srgb,
};

enum class BCEncodeQuality { fastest, normal, production, highest };

struct BCEncodeOptions {
    BCEncodeQuality quality = BCEncodeQuality::normal;
    bool generate_mipmaps = false;
    MipFilter mip_filter = MipFilter::box;  // MipFilter defined in bitmap.h
    uint32_t channel_weights[4] = {1, 1, 1, 1}; // BC7 only: RGBA error weights passed to bc7enc_compress_block_params::m_weights. Ignored by rgbcx (BC1-5).
    bool has_alpha = true;       // BC7 only: if false, bc7enc sets m_force_alpha=false allowing more bits for color. Ignored by other formats.
    bool prefer_nvtt = true;     // Use NVTT3 if available
};

struct BCCompressedMip {
    uint32_t width, height;
    std::vector<uint8_t> data;
};

struct BCCompressedImage {
    BCFormat format;
    std::vector<BCCompressedMip> mip_levels;
};
```

Provide `BCFormat`↔`Format` conversion helpers (e.g., `bc_format_to_format()`, `format_to_bc_format()`) so callers can interop with the RHI `Format` enum when needed.

Utility functions (in `bc_types.h`):
- `bc_format_bytes_per_block(BCFormat) -> uint32_t` — returns 8 (BC1/BC4) or 16 (BC2/BC3/BC5/BC6H/BC7)
- `bc_compressed_size(uint32_t width, uint32_t height, BCFormat format) -> size_t` — total bytes for one mip level: `((width + 3) / 4) * ((height + 3) / 4) * bytes_per_block`. Note: uses ceiling division, not truncating `>>2` (unlike bcdec's macros which are wrong for non-multiple-of-4 sizes).
- `bc_mip_count(uint32_t width, uint32_t height) -> uint32_t` — full mipchain length: `floor(log2(max(w,h))) + 1`

sRGB format variants (e.g., `bc1_unorm_srgb`) are metadata only — encoding/decoding operates on raw byte values. The sRGB tag is preserved in `BCCompressedImage::format` and passed through to GPU texture creation, but no gamma conversion is performed.

## 3c. BCCodec Class

Defined in `src/sgl/core/bc_codec.h`:

```
class SGL_API BCCodec {
public:
    BCCodec();
    ~BCCodec();

    // Non-copyable, non-movable
    BCCodec(const BCCodec&) = delete;
    BCCodec& operator=(const BCCodec&) = delete;

    // Encoding
    BCCompressedImage encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options = {});

    // Decoding
    void decode(const void* data, size_t size, BCFormat format,
                uint32_t width, uint32_t height, const BCMutableImage& dst);

    // Query
    bool is_nvtt_available() const;
    bool can_encode(BCFormat format) const;
    bool can_decode(BCFormat format) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl; // Holds NVTT3 state (lazy-loaded)
};
```

## 3d. SW Backend Implementation (`bc_codec.cpp`)

**Block size constants** (defined in `bc_codec.cpp`):
- BC1/BC4: 8 bytes per 4×4 block
- BC2/BC3/BC5/BC6H/BC7: 16 bytes per 4×4 block

**Decoding (bcdec):**

`BCDEC_BC4BC5_PRECISE` must be defined before including `bcdec.h` to enable signed (snorm) BC4/BC5 decoding. Without this define, the `isSigned` parameter is unavailable.

- For each 4×4 block: call the appropriate `bcdec_bcN()` function
- Handle non-multiple-of-4 dimensions by decoding full blocks and clipping
- Decode output format matches the compressed format's defined pixel layout:
  - BC1: RGBA uint8 (4 channels)
  - BC2: RGBA uint8 (4 channels)
  - BC3: RGBA uint8 (4 channels)
  - BC4: R uint8 (1 channel); for snorm, use `bcdec_bc4(..., isSigned=1)`
  - BC5: RG uint8 (2 channels); for snorm, use `bcdec_bc5(..., isSigned=1)`
  - BC6H: RGB float16 (3 channels) — use `bcdec_bc6h_half()` which is lossless (BC6H natively stores half-precision endpoints). Avoids unnecessary half→float→half round-trip.
  - BC7: RGBA uint8 (4 channels)

**Encoding (rgbcx + bc7enc):**
- `rgbcx::init()` and `bc7enc_compress_block_init()` called once, guarded by `std::call_once`
- For each 4×4 block: extract pixels, pad if necessary, call appropriate encoder
- BC1: `rgbcx::encode_bc1(level, pDst, pPixels, allow_3color=false, use_transparent_texels_for_black=false)`
- BC2: custom (quantize alpha to 4 bits) + `rgbcx::encode_bc1()` for color
- BC3: `rgbcx::encode_bc3()`
- BC4: `rgbcx::encode_bc4()`
- BC5: `rgbcx::encode_bc5()`
- BC6H: error — requires NVTT3
- BC7: `bc7enc_compress_block()`

`BCCodec::encode()` with `generate_mipmaps=true` calls `Bitmap::generate_mip_chain()` (from Phase 2) on the source, then encodes each level.

## 3e. CMake Integration

- Add `core/bc_types.h`, `core/bc_codec.h` and `core/bc_codec.cpp` to `src/sgl/CMakeLists.txt` source list
- Link `sgl` against `bcdec` (INTERFACE) and `bc7enc` (PRIVATE)

**Relevant files:**
- `src/sgl/CMakeLists.txt` — Add source files and link targets
- `src/sgl/core/bc_types.h` — Type definitions (new)
- `src/sgl/core/bc_codec.h` — BCCodec class declaration (new)
- `src/sgl/core/bc_codec.cpp` — Implementation (new)

## 3f. C++ Tests

Create `tests/sgl/core/test_bc_codec.cpp` following the existing doctest pattern (see `test_dds_file.cpp`).

**Test cases:**
1. **Utility functions** — `bc_format_bytes_per_block`, `bc_compressed_size`, `bc_mip_count` for various inputs
2. **BCFormat↔Format conversion** — round-trip all BCFormat values through `bc_format_to_format` / `format_to_bc_format`
3. **Roundtrip per format** — For each SW-supported format (BC1, BC2, BC3, BC4, BC5, BC7): encode a synthetic 4×4 block, decode it, verify output dimensions and basic value sanity
4. **Roundtrip larger image** — Encode/decode a 64×64 gradient image per format, compute PSNR and verify it exceeds a minimum threshold
5. **Non-multiple-of-4 sizes** — Encode a 13×7 image, verify block padding handled correctly
6. **Small images** — 1×1, 2×2, 3×3, 4×4
7. **Encode with mipmaps** — Encode 64×64 with `generate_mipmaps=true`, verify correct level count (7) and per-level compressed dimensions
8. **Quality levels** — Encode at fastest/highest, verify both produce valid output
9. **Channel weights** — Encode BC7 with non-uniform weights, verify no crash (quality difference is visual; other formats ignore weights)
10. **has_alpha hint** — Encode BC7 with `has_alpha=false`, verify valid output (only meaningful for BC7)
11. **can_encode/can_decode** — Verify correct reporting (BC6H encode → false without NVTT3, all decode → true)
12. **BC6H encode error** — Verify encoding BC6H without NVTT3 raises an appropriate error
13. **Decode output format** — Verify BC4→1ch uint8, BC5→2ch uint8, BC6H→3ch float16, others→4ch RGBA uint8

**Relevant files:**
- `tests/sgl/core/test_bc_codec.cpp` (new)
- `tests/CMakeLists.txt` — New test file must be explicitly added to `target_sources(sgl_tests ...)` (tests are NOT auto-discovered)
