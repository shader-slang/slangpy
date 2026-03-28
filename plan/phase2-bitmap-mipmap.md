# Phase 2: Bitmap Resample & Mipmap Generation

Add CPU image resampling and mipmap generation to `Bitmap`. This is independent of BC codec and external deps — can be done in parallel with Phase 1.

## 2a. ResamplingFilter Types

Add to `src/sgl/core/bitmap.h`:

```cpp
struct BoxFilter {};

struct KaiserFilter {
    float alpha = 4.0f;
    float width = 3.0f;
};

struct MitchellFilter {
    float b = 1.0f / 3.0f;
    float c = 1.0f / 3.0f;
};

using ResamplingFilter = std::variant<BoxFilter, KaiserFilter, MitchellFilter>;
```

Uses `std::variant` so each filter type carries only its own parameters. Adding a new filter later is just a new struct + variant alternative + `std::visit` case — the compiler catches any unhandled filters.

## 2b. Resample & Mipmap Methods

Add to `Bitmap`:
```cpp
/// Resample to arbitrary resolution using a separable filter.
/// Boundary condition: clamp (samples outside the image edge repeat the border pixel).
ref<Bitmap> resample(uint32_t width, uint32_t height,
                     ResamplingFilter filter = BoxFilter{}) const;

/// Convenience: resample to half resolution (rounding down, min 1×1).
ref<Bitmap> generate_mip(ResamplingFilter filter = BoxFilter{}) const;

/// Convenience: generate full mip chain from next level down to 1×1 (excludes source).
std::vector<ref<Bitmap>> generate_mip_chain(ResamplingFilter filter = BoxFilter{}) const;
```

`generate_mip()` and `generate_mip_chain()` are thin wrappers over `resample()`:
```cpp
ref<Bitmap> Bitmap::generate_mip(ResamplingFilter filter) const {
    return resample(std::max(1u, width() / 2), std::max(1u, height() / 2), filter);
}
```

Call-site examples:
```cpp
// Resample to arbitrary size
bitmap->resample(256, 256);                                   // BoxFilter{}
bitmap->resample(100, 50, MitchellFilter{});                  // Mitchell 1/3,1/3

// Mip generation
bitmap->generate_mip();                                       // BoxFilter{}
bitmap->generate_mip(KaiserFilter{});                         // Kaiser with defaults
bitmap->generate_mip(MitchellFilter{.b = 0.5f, .c = 0.5f});  // Custom Mitchell
bitmap->generate_mip_chain(KaiserFilter{.alpha = 3.0f});      // Custom Kaiser chain
```

### Resampling implementation

The `resample()` method uses a separable 2-pass approach (horizontal then vertical), similar to Mitsuba 3's `Resampler`:
1. For each output row/column, compute which source samples contribute based on the filter radius
2. Precompute normalized weights for each tap
3. Apply horizontally (all rows), writing to an intermediate buffer
4. Apply vertically (all columns), writing to the output

Boundary condition is clamp (edge pixels are repeated). This is the correct default for textures and the only mode needed for BC encoding. Additional boundary modes (repeat, mirror, zero) can be added later.

Filter dispatch via `std::visit`:
```cpp
std::visit(overloaded{
    [&](const BoxFilter&)       { /* radius=0.5, eval=1 inside */ },
    [&](const KaiserFilter& f)  { /* use f.alpha, f.width */ },
    [&](const MitchellFilter& f) { /* use f.b, f.c, radius=2 */ },
}, filter);
```

When downsampling, the filter radius is scaled by `src_res / dst_res` (low-pass filter scaling), matching Mitsuba's approach.

### Behavior

- `resample()` — resize to any target resolution with the given filter
- `generate_mip()` — returns a new `Bitmap` at half resolution (rounding down, min 1×1)
- `generate_mip_chain()` — returns the full chain from the next level down to 1×1 (does NOT include the source level)
- Box filter: radius 0.5, uniform weight — for mips, this is equivalent to averaging 2×2 pixel neighborhoods
- Kaiser filter: wider kernel with Kaiser window (precomputed weights); `alpha` controls sidelobe attenuation, `width` controls kernel radius in pixels
- Mitchell filter: Mitchell-Netravali bicubic (radius 2); `b` and `c` control the tradeoff between blurring and ringing (default 1/3, 1/3)
- Supports all `Bitmap::ComponentType` values (uint8, float16, float32, etc.)
- **sRGB linearization**: if `srgb_gamma() == true` and the component type is uint8, the implementation internally converts to linear float32, resamples, then converts back to sRGB uint8. This ensures correct filtering — averaging in sRGB space is mathematically wrong because the transfer function is nonlinear (dark regions get biased darker). For float formats, data is assumed already linear and resampled directly. This matches Mitsuba 3's expectation (which only resamples float data, requiring manual conversion), but is more convenient since our primary use case is sRGB uint8 source images.

**Relevant files:**
- `src/sgl/core/bitmap.h` — Add `ResamplingFilter` variant type, `resample()`, and mip method declarations
- `src/sgl/core/bitmap.cpp` — Add resampling and mipmap generation implementations

## 2c. Mipmap Tests

Create `tests/sgl/core/test_bitmap_mipmap.cpp` following the existing doctest pattern.

**Test cases:**
**Resample tests:**
1. **resample identity** — resample to same resolution, verify output matches input
2. **resample arbitrary downscale** — 256×256 → 100×100, verify output dimensions and no crash
3. **resample arbitrary upscale** — 64×64 → 256×256, verify output dimensions and no crash
4. **resample non-square** — 200×100 → 50×25 with MitchellFilter, verify dimensions
5. **resample box correctness** — 4×4 solid color → 2×2, verify all pixels match source color

**Mip generation tests:**
6. **generate_mip dimensions** — 64×64→32×32, 63×63→31×31, 1×1→1×1, 4×3→2×1
7. **generate_mip_chain length** — 64×64 produces 6 levels (32², 16², 8², 4², 2², 1×1)
8. **generate_mip_chain dimensions** — verify each level is half the previous (rounding down)
9. **Box filter correctness** — 2×2 solid color → 1×1 same color; 2×2 black/white checkerboard → 1×1 mid-gray
10. **All filters produce valid output** — `BoxFilter{}`, `KaiserFilter{}`, `MitchellFilter{}` on 64×64 uint8 RGBA, verify no crashes and correct dimensions
11. **Custom filter parameters** — `KaiserFilter{.alpha = 3.0f, .width = 2.0f}`, `MitchellFilter{.b = 0.5f, .c = 0.5f}` produce valid output
12. **Non-power-of-2** — 100×60 mip chain, verify all levels have correct dimensions
13. **1×N and N×1 images** — verify chain terminates at 1×1

**Component & format tests:**
14. **Component types** — verify resample works for uint8, float16, float32
15. **Channel counts** — verify resample works for 1, 2, 3, 4 channel images
16. **sRGB flag preservation** — verify `srgb_gamma()` is preserved across mip levels
17. **sRGB linearization correctness** — create a 2×2 sRGB uint8 bitmap with known values (e.g., 0 and 255), generate_mip with BoxFilter, verify the 1×1 result matches the linear-space average re-encoded to sRGB (~188, not 128). Compare against naive sRGB-space average to confirm they differ.
18. **Float32 linear passthrough** — verify that a float32 bitmap with `srgb_gamma() == false` is resampled directly without any gamma transformation

**Relevant files:**
- `tests/sgl/core/test_bitmap_mipmap.cpp` (new)
- `tests/CMakeLists.txt` — Add to `target_sources(sgl_tests ...)` (tests are NOT auto-discovered)
