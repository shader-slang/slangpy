# Phase 2: Bitmap Mipmap Generation

Add CPU mipmap generation to `Bitmap`. This is independent of BC codec and external deps — can be done in parallel with Phase 1.

## 2a. MipFilter Enum

Add to `src/sgl/core/bitmap.h`:

```
enum class MipFilter { box, kaiser, mitchell };
```

## 2b. Mipmap Methods

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

## 2c. Mipmap Tests

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
