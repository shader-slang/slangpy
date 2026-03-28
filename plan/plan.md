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

## Phases

| Phase | Description | Details |
|-------|-------------|---------|
| 1 | External Dependencies | [phase1-external-deps.md](plan/phase1-external-deps.md) |
| 2 | Bitmap Resample & Mipmap Generation | [phase2-bitmap-mipmap.md](plan/phase2-bitmap-mipmap.md) |
| 3 | BC Codec Implementation | [phase3-bc-codec.md](plan/phase3-bc-codec.md) |
| 4 | Finalize | [phase4-finalize.md](plan/phase4-finalize.md) |
| 5 | NVTT3 Dynamic Loading | [phase5-nvtt3.md](plan/phase5-nvtt3.md) |

## Dependency Graph

```
Phase 1 (external deps + licensing)    Phase 2 (Bitmap resample + mipmap + tests)
              ↓                                     ↓
Phase 3a-3b (bc_types.h) → Phase 3c (bc_codec.h) → Phase 3d (bc_codec.cpp) → Phase 3e (CMake) → Phase 3f (C++ tests)
  ↓
Phase 4 (finalize) — build + test + pre-commit
  ↓
Phase 5 (NVTT3) — optional, adds accelerated backend + BC6H encode
```

Phases 1 and 2 are independent and can be done in parallel. Phase 3 depends on both.

## Relevant files (summary)

- `external/include/bcdec.h` (new, vendored) — Header-only BC decoder (no CMake target, already on include path)
- `external/bc7enc/rgbcx.h`, `rgbcx.cpp`, `bc7enc.h`, `bc7enc.cpp` (new, vendored) — SW BC encoders
- `external/CMakeLists.txt` — Add bc7enc static library target
- `src/sgl/core/bc_types.h` (new) — BCFormat, BCImage, BCMutableImage, BCEncodeOptions, BCCompressedImage, utility functions
- `src/sgl/core/bc_codec.h` (new) — BCCodec class declaration
- `src/sgl/core/bc_codec.cpp` (new) — SW backend (NVTT3 dynamic loading added in Phase 5)
- `src/sgl/core/bitmap.h` — Add `ResamplingFilter` variant type (`BoxFilter`, `KaiserFilter`, `MitchellFilter`), `resample()` / `generate_mip()` / `generate_mip_chain()` declarations
- `src/sgl/core/bitmap.cpp` — Add resampling and mipmap generation implementations
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
- **Resample + mipmap generation lives in `Bitmap`** — `resample()` is the general-purpose method; `generate_mip()` / `generate_mip_chain()` are thin wrappers. Separable 2-pass implementation (horizontal then vertical) with precomputed weights, similar to Mitsuba 3's `Resampler`. Boundary condition is clamp only (sufficient for textures and BC encoding; more modes can be added later). Implemented in Phase 2 with dedicated tests.
- **`ResamplingFilter` is a `std::variant<BoxFilter, KaiserFilter, MitchellFilter>`** — each filter struct carries its own parameters (e.g., `MitchellFilter{.b, .c}`, `KaiserFilter{.alpha, .width}`). Type-safe, extensible, dispatched via `std::visit`. Default is `BoxFilter{}`. Chosen over an enum (no way to pass parameters) and a tagged struct (fields meaningless for wrong filter type).
- **Mipmap filters** — box (fast default), Kaiser (higher quality, configurable `alpha`/`width`), Mitchell (good general-purpose, configurable `b`/`c`)
- **Channel weights** — `uint32_t[4]` in `BCEncodeOptions`, mapped to `bc7enc_compress_block_params::m_weights`. Only used for BC7 encoding. Ignored for BC1-5 (rgbcx does not support per-channel weights).
- **Alpha hint** — `has_alpha` in `BCEncodeOptions`; only affects BC7 encoding (`bc7enc_compress_block_params::m_force_alpha`). Ignored for other formats.
- **Utility functions in `bc_types.h`** — `bc_format_bytes_per_block`, `bc_compressed_size`, `bc_mip_count`
- **`bc_compressed_size` uses ceiling division** — `((w+3)/4) * ((h+3)/4) * bpb`, not `>>2` (bcdec's macros truncate, which is wrong for non-multiple-of-4 sizes)
- **`BCFormat` enum** defined in `bc_types.h` — avoids core→device dependency; conversion helpers provided for interop with RHI `Format`
- **sRGB is metadata only (BC codec)** — no gamma conversion performed in BC encode/decode; sRGB tag preserved in BCFormat and passed through to texture creation
- **sRGB linearization in mipmap generation** — `generate_mip()`/`generate_mip_chain()` automatically linearize sRGB uint8 data before filtering and re-encode to sRGB afterward. Filtering in sRGB space produces incorrect results (nonlinear transfer function biases dark regions). Float formats are assumed already linear. Inspired by Mitsuba 3, which requires float input for resampling — we handle the conversion internally for convenience.
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
