# Plan: BC Texture Encoder/Decoder

Add CPU-based BC1-7 block compression encoding and decoding to SGL, with a software backend (bcdec/bc7enc/rgbcx) and an optional high-quality backend via dynamically-loaded NVTT3.

## TL;DR

New `BCCodec` class in `sgl/core/bc_codec.h/.cpp` providing encode/decode for BC1-7 formats. Types defined in `sgl/core/bc_types.h` (separated for reuse by `Bitmap` and other consumers). Decoding uses bcdec (header-only). Encoding uses rgbcx (BC1-5) + bc7enc (BC7) for SW, with NVTT3 loaded at runtime as an optional accelerated backend for all formats including BC6H. CPU mipmap generation (box, Kaiser, and Mitchell filters) is added to `Bitmap` in a standalone phase with its own tests. C++ tests cover the codec.

## Format Support Matrix

| Format   | SW Encode      | SW Decode | NVTT3 Encode | Notes                              |
|----------|----------------|-----------|--------------|------------------------------------|
| BC1      | ✓ rgbcx        | ✓ bcdec   | ✓            |                                    |
| BC2      | ✓ custom+rgbcx | ✓ bcdec   | ✓            | 4-bit explicit alpha + BC1 color   |
| BC3      | ✓ rgbcx        | ✓ bcdec   | ✓            |                                    |
| BC4      | ✓ rgbcx        | ✓ bcdec   | ✓            |                                    |
| BC5      | ✓ rgbcx        | ✓ bcdec   | ✓            |                                    |
| BC6H     | ✗              | ✓ bcdec   | ✓            | Requires NVTT3 for encoding        |
| BC7      | ✓ bc7enc       | ✓ bcdec   | ✓            |                                    |

## Phase 1: External Dependencies

Add bcdec, rgbcx, and bc7enc as vendored external dependencies.

1. **Add bcdec** — Copy `bcdec.h` (single header, MIT/public domain) into `external/bcdec/`. Create a minimal CMake target (interface library, header-only). Add license file.
2. **Add bc7enc_rdo files** — Copy `rgbcx.h`, `rgbcx.cpp`, `bc7enc.h`, `bc7enc.cpp` into `external/bc7enc/`. Create a static library CMake target. Disable warnings as needed (follow the `lmdb` pattern in `external/CMakeLists.txt`). Add license file.
3. **Update `external/CMakeLists.txt`** — Add the two new subdirectories/targets.
4. **Update `LICENSES/`** — Add license texts for bcdec (MIT) and bc7enc_rdo (MIT). Verify exact license terms from upstream repos before vendoring.

**Relevant files:**
- `external/CMakeLists.txt` — Follow the `lmdb` pattern: `add_library(bcdec INTERFACE)` + `target_include_directories(bcdec INTERFACE bcdec)`, and `add_library(bc7enc STATIC)` + source list + warning suppression.

## Phase 2: Bitmap Mipmap Generation

Add CPU mipmap generation to `Bitmap`. This is independent of BC codec and external deps — can be done in parallel with Phase 1.

### 2a. MipFilter Enum

Add to `src/sgl/core/bitmap.h`:

```
enum class MipFilter { box, kaiser, mitchell };
```

### 2b. Mipmap Methods

Add to `Bitmap`:
```
ref<Bitmap> generate_mip(MipFilter filter = MipFilter::box) const;
std::vector<ref<Bitmap>> generate_mip_chain(MipFilter filter = MipFilter::box) const;
```

- `generate_mip()` — returns a new `Bitmap` at half resolution (rounding down, min 1×1)
- `generate_mip_chain()` — returns the full chain from the next level down to 1×1 (does NOT include the source level)
- Box filter: average 2×2 pixel neighborhoods
- Kaiser filter: wider kernel with Kaiser window (precomputed weights)
- Mitchell filter: Mitchell-Netravali bicubic (B=1/3, C=1/3) — good general-purpose default
- Supports all `Bitmap::ComponentType` values (uint8, float16, float32, etc.)
- Handles odd dimensions correctly (last row/column averaged with fewer samples)

**Relevant files:**
- `src/sgl/core/bitmap.h` — Add `MipFilter` enum and method declarations
- `src/sgl/core/bitmap.cpp` — Add mipmap generation implementations

### 2c. Mipmap Tests

Create `tests/sgl/core/test_bitmap_mipmap.cpp` following the existing doctest pattern.

**Test cases:**
1. **generate_mip dimensions** — 64×64→32×32, 63×63→31×31, 1×1→1×1, 4×3→2×1
2. **generate_mip_chain length** — 64×64 produces 6 levels (32², 16², 8², 4², 2², 1×1)
3. **generate_mip_chain dimensions** — verify each level is half the previous (rounding down)
4. **Box filter correctness** — 2×2 solid color → 1×1 same color; 2×2 black/white checkerboard → 1×1 mid-gray
5. **All filters produce valid output** — box, Kaiser, Mitchell on 64×64 uint8 RGBA, verify no crashes and correct dimensions
6. **Component types** — verify mipmap works for uint8, float16, float32
7. **Channel counts** — verify mipmap works for 1, 2, 3, 4 channel images
8. **Non-power-of-2** — 100×60 mip chain, verify all levels have correct dimensions
9. **1×N and N×1 images** — verify chain terminates at 1×1
10. **sRGB flag preservation** — verify `srgb_gamma()` is preserved across mip levels

**Relevant files:**
- `tests/sgl/core/test_bitmap_mipmap.cpp` (new)
- `tests/CMakeLists.txt` — Add to `target_sources(sgl_tests ...)` (tests are NOT auto-discovered)

## Phase 3: BC Codec Implementation

### 3a. Image Types

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

### 3b. Encode/Decode Options & Result Types

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

### 3c. BCCodec Class

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

### 3d. SW Backend Implementation (`bc_codec.cpp`)

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

### 3e. CMake Integration

- Add `core/bc_types.h`, `core/bc_codec.h` and `core/bc_codec.cpp` to `src/sgl/CMakeLists.txt` source list
- Link `sgl` against `bcdec` (INTERFACE) and `bc7enc` (PRIVATE)


**Relevant files:**
- `src/sgl/CMakeLists.txt` — Add source files and link targets
- `src/sgl/core/bc_types.h` — Type definitions (new)
- `src/sgl/core/bc_codec.h` — BCCodec class declaration (new)
- `src/sgl/core/bc_codec.cpp` — Implementation (new)

### 3f. C++ Tests

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

## Phase 4: Finalize

1. **Run pre-commit** — `pre-commit run --all-files`
2. **Build and test** — `cmake --build --preset windows-msvc-debug && python tools/ci.py unit-test-cpp`
3. `pre-commit run --all-files` — passes (re-run if it modifies files)

Note: No Python bindings in this phase. The BC codec is C++ only.

## Phase 5: NVTT3 Dynamic Loading

Optional accelerated backend for all formats including BC6H encoding. Added after SW path is working end-to-end.

### 5a. NVTT3 Dynamic Loading (`bc_codec.cpp`, private implementation)

Follow the RenderDoc dynamic loading pattern from `src/sgl/utils/renderdoc.cpp`:

- Define function pointer types for the NVTT3 C API functions we need (subset of nvtt_wrapper.h)
- Key functions to load:
  - `nvttCreateCPUInputBuffer`, `nvttDestroyCPUInputBuffer`
  - `nvttEncodeCPU` (unified encode, new API)
  - `nvttCreateSurface`, `nvttDestroySurface`, `nvttSurfaceSetImageData`
  - `nvttSurfaceBuildNextMipmapDefaults` (for mipmap generation)
  - `nvttSurfaceData`, `nvttSurfaceWidth`, `nvttSurfaceHeight`
  - `nvttVersion`
- Library search order:
  1. Next to sgl.dll / libsgl.so (same directory)
  2. System PATH / LD_LIBRARY_PATH
  3. Optional: `SGL_NVTT_PATH` environment variable
- DLL names: `nvtt.dll` (Windows), `libnvtt.so` (Linux), `libnvtt.dylib` (macOS)
- Lazy initialization: load on first use, guarded by `std::call_once` for thread safety
- If NVTT3 not found, fall back to SW silently (log info message)

**NVTT3 encode flow:**
1. Convert `BCImage` to `NvttRefImage`
2. Create `NvttCPUInputBuffer` from ref image
3. Allocate output buffer (size = blocks × bytes_per_block)
4. Call `nvttEncodeCPU()` with appropriate `NvttEncodeSettings`
5. For mipmaps: create `NvttSurface`, call `nvttSurfaceBuildNextMipmapDefaults()` for each level, encode each

**Mapping BCFormat → NvttFormat:**
- `bc1_unorm` / `bc1_unorm_srgb` → `NVTT_Format_BC1`
- `bc2_unorm` / `bc2_unorm_srgb` → `NVTT_Format_BC2`
- `bc3_unorm` / `bc3_unorm_srgb` → `NVTT_Format_BC3`
- `bc4_unorm` → `NVTT_Format_BC4`, `bc4_snorm` → `NVTT_Format_BC4S`
- `bc5_unorm` → `NVTT_Format_BC5`, `bc5_snorm` → `NVTT_Format_BC5S`
- `bc6h_ufloat` → `NVTT_Format_BC6U`, `bc6h_sfloat` → `NVTT_Format_BC6S`
- `bc7_unorm` / `bc7_unorm_srgb` → `NVTT_Format_BC7`

### 5b. NVTT3 Tests

- **NVTT3 backend** (conditional) — If NVTT3 is available, test BC6H encoding, test that NVTT3 backend produces valid output for all formats
- Add conditional C++ test cases in `test_bc_codec.cpp` that check `is_nvtt_available()` and skip if unavailable

## Dependency Graph

```
Phase 1 (external deps + licensing)    Phase 2 (Bitmap mipmap + tests)
              ↓                                     ↓
Phase 3a-3b (bc_types.h) → Phase 3c (bc_codec.h) → Phase 3d (bc_codec.cpp) → Phase 3e (CMake) → Phase 3f (C++ tests)
  ↓
Phase 4 (finalize) — build + test + pre-commit
  ↓
Phase 5 (NVTT3) — optional, adds accelerated backend + BC6H encode
```

Phases 1 and 2 are independent and can be done in parallel. Phase 3 depends on both.

## Relevant files (summary)

- `external/bcdec/bcdec.h` (new, vendored) — Header-only BC decoder
- `external/bc7enc/rgbcx.h`, `rgbcx.cpp`, `bc7enc.h`, `bc7enc.cpp` (new, vendored) — SW BC encoders
- `external/CMakeLists.txt` — Add bcdec + bc7enc targets
- `LICENSES/` — Add license texts for bcdec and bc7enc_rdo
- `src/sgl/core/bc_types.h` (new) — BCFormat, BCImage, BCMutableImage, BCEncodeOptions, BCCompressedImage, utility functions
- `src/sgl/core/bc_codec.h` (new) — BCCodec class declaration
- `src/sgl/core/bc_codec.cpp` (new) — SW backend (NVTT3 dynamic loading added in Phase 5)
- `src/sgl/core/bitmap.h` — Add `MipFilter` enum, `generate_mip()` / `generate_mip_chain()` declarations
- `src/sgl/core/bitmap.cpp` — Add mipmap generation implementations
- `src/sgl/CMakeLists.txt` — Add bc_types.h/bc_codec.h/bc_codec.cpp, link bcdec/bc7enc
- `tests/sgl/core/test_bitmap_mipmap.cpp` (new) — Bitmap mipmap unit tests (doctest)
- `tests/sgl/core/test_bc_codec.cpp` (new) — BC codec unit tests (doctest)
- `tests/CMakeLists.txt` — Add both test files to `target_sources(sgl_tests ...)`
- `src/sgl/device/formats.h` — Referenced for `Format` enum (BCFormat↔Format conversion helpers)
- `src/sgl/core/platform.h` — Referenced for `load_shared_library()` (used in Phase 5)

## Verification

1. `cmake --preset windows-msvc --fresh && cmake --build --preset windows-msvc-debug` — builds cleanly
2. `python tools/ci.py unit-test-cpp` — C++ tests pass (including test_bitmap_mipmap and test_bc_codec)
3. Without NVTT3 DLL: BC1-5/BC7 encode+decode works, BC6H encode raises clear error
4. With NVTT3 DLL placed alongside sgl: all formats work, `is_nvtt_available` returns True
5. `pre-commit run --all-files` — passes (re-run if it modifies files)
6. Roundtrip quality: decoded images are within expected PSNR bounds for each format

## Decisions

- **bc7enc** (not "bc8enc") confirmed as the intended BC7 encoder library
- **BC1a dropped** — no distinct `BCFormat` for BC1a. BC1a is not a separate GPU format (no `Format::bc1a_*` in the RHI). The BC1 format inherently supports the 3-color + transparent black mode; this is a decode-time interpretation, not an encode-time format choice.
- **BC6H encoding requires NVTT3** — no SW fallback (would need Compressonator or custom impl)
- **BC2 SW encoding** — custom implementation composing rgbcx BC1 + explicit 4-bit alpha quantization
- **Mipmap generation lives in `Bitmap`** — `generate_mip()` / `generate_mip_chain()` methods on `Bitmap`, reusable beyond BC encoding. Implemented in its own phase (Phase 2) with dedicated tests.
- **`MipFilter` enum in `bitmap.h`** — not BC-specific; renamed from `BCMipFilter` since it's a Bitmap feature
- **Mipmap filters** — box (fast default), Kaiser (higher quality), Mitchell (good general-purpose)
- **Channel weights** — `uint32_t[4]` in `BCEncodeOptions`, mapped to `bc7enc_compress_block_params::m_weights`. Only used for BC7 encoding. Ignored for BC1-5 (rgbcx does not support per-channel weights).
- **Alpha hint** — `has_alpha` in `BCEncodeOptions`; only affects BC7 encoding (`bc7enc_compress_block_params::m_force_alpha`). Ignored for other formats.
- **Utility functions in `bc_types.h`** — `bc_format_bytes_per_block`, `bc_compressed_size`, `bc_mip_count`
- **`bc_compressed_size` uses ceiling division** — `((w+3)/4) * ((h+3)/4) * bpb`, not `>>2` (bcdec's macros truncate, which is wrong for non-multiple-of-4 sizes)
- **`BCFormat` enum** defined in `bc_types.h` — avoids core→device dependency; conversion helpers provided for interop with RHI `Format`
- **sRGB is metadata only** — no gamma conversion performed; sRGB tag preserved in BCFormat and passed through to texture creation
- **Decode output matches format** — BC4→1ch uint8, BC5→2ch uint8, BC6H→3ch float16 (via `bcdec_bc6h_half()`, lossless), others→4ch RGBA uint8
- **`BCComponentType` = `DataStruct::Type`** — reuses the existing type enum from `data_struct.h` (same as Bitmap's ComponentType), not `DataType` from `data_type.h`
- **`BCCodec` does not inherit from `Object`** — uses pimpl pattern (`std::unique_ptr<Impl>`) for NVTT3 state. No reference counting needed.
- **`BCDEC_BC4BC5_PRECISE` must be defined** — required for signed (snorm) BC4/BC5 decoding support
- **All init guarded by `std::call_once`** — `rgbcx::init()`, `bc7enc_compress_block_init()`, and NVTT3 lazy loading are all thread-safe
- **No Bitmap overloads on BCCodec** — only `BCImage`-based API; callers use `bc_image_from_bitmap()` helper
- **No Python bindings** — C++ only; Python bindings may be added later as a separate task
- **`BCImage`/`BCMutableImage`** — lightweight non-owning views, not exposed outside C++
- **File split: `bc_types.h` + `bc_codec.h`/`bc_codec.cpp`** — types separated for reuse by `Bitmap` and other consumers without pulling in codec implementation
- **Vendored library licenses** — bcdec (MIT), bc7enc_rdo (MIT) — license texts added to `LICENSES/`
- **NVTT3 is encoding-only** — decoding always uses bcdec (simpler, no DLL required)
- **No NVTT3 header shipped** — all NVTT3 types/function pointers defined locally in `bc_codec.cpp` (private implementation detail)
