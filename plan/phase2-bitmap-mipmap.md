# Phase 2: Bitmap Mipmap Generation

Add CPU mipmap generation to `Bitmap`. This is independent of BC codec and external deps — can be done in parallel with Phase 1.

## 2a. MipFilter Types

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

using MipFilter = std::variant<BoxFilter, KaiserFilter, MitchellFilter>;
```

Uses `std::variant` so each filter type carries only its own parameters. Adding a new filter later is just a new struct + variant alternative + `std::visit` case — the compiler catches any unhandled filters.

## 2b. Mipmap Methods

Add to `Bitmap`:
```cpp
ref<Bitmap> generate_mip(MipFilter filter = BoxFilter{}) const;
std::vector<ref<Bitmap>> generate_mip_chain(MipFilter filter = BoxFilter{}) const;
```

Call-site examples:
```cpp
bitmap->generate_mip();                                       // BoxFilter{}
bitmap->generate_mip(KaiserFilter{});                         // Kaiser with defaults
bitmap->generate_mip(MitchellFilter{.b = 0.5f, .c = 0.5f});  // Custom Mitchell
bitmap->generate_mip_chain(KaiserFilter{.alpha = 3.0f});      // Custom Kaiser chain
```

Implementation dispatches via `std::visit`:
```cpp
std::visit(overloaded{
    [&](const BoxFilter&)       { /* box impl */ },
    [&](const KaiserFilter& f)  { /* use f.alpha, f.width */ },
    [&](const MitchellFilter& f) { /* use f.b, f.c */ },
}, filter);
```

- `generate_mip()` — returns a new `Bitmap` at half resolution (rounding down, min 1×1)
- `generate_mip_chain()` — returns the full chain from the next level down to 1×1 (does NOT include the source level)
- Box filter: average 2×2 pixel neighborhoods
- Kaiser filter: wider kernel with Kaiser window (precomputed weights); `alpha` controls sidelobe attenuation, `width` controls kernel radius in pixels
- Mitchell filter: Mitchell-Netravali bicubic; `b` and `c` control the tradeoff between blurring and ringing (default 1/3, 1/3)
- Supports all `Bitmap::ComponentType` values (uint8, float16, float32, etc.)
- Handles odd dimensions correctly (last row/column averaged with fewer samples)

**Relevant files:**
- `src/sgl/core/bitmap.h` — Add `MipFilter` variant type and method declarations
- `src/sgl/core/bitmap.cpp` — Add mipmap generation implementations

## 2c. Mipmap Tests

Create `tests/sgl/core/test_bitmap_mipmap.cpp` following the existing doctest pattern.

**Test cases:**
1. **generate_mip dimensions** — 64×64→32×32, 63×63→31×31, 1×1→1×1, 4×3→2×1
2. **generate_mip_chain length** — 64×64 produces 6 levels (32², 16², 8², 4², 2², 1×1)
3. **generate_mip_chain dimensions** — verify each level is half the previous (rounding down)
4. **Box filter correctness** — 2×2 solid color → 1×1 same color; 2×2 black/white checkerboard → 1×1 mid-gray
5. **All filters produce valid output** — `BoxFilter{}`, `KaiserFilter{}`, `MitchellFilter{}` on 64×64 uint8 RGBA, verify no crashes and correct dimensions
6. **Custom filter parameters** — `KaiserFilter{.alpha = 3.0f, .width = 2.0f}`, `MitchellFilter{.b = 0.5f, .c = 0.5f}` produce valid output
7. **Component types** — verify mipmap works for uint8, float16, float32
8. **Channel counts** — verify mipmap works for 1, 2, 3, 4 channel images
9. **Non-power-of-2** — 100×60 mip chain, verify all levels have correct dimensions
10. **1×N and N×1 images** — verify chain terminates at 1×1
11. **sRGB flag preservation** — verify `srgb_gamma()` is preserved across mip levels

**Relevant files:**
- `tests/sgl/core/test_bitmap_mipmap.cpp` (new)
- `tests/CMakeLists.txt` — Add to `target_sources(sgl_tests ...)` (tests are NOT auto-discovered)
