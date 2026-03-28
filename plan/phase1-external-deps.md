# Phase 1: External Dependencies

Add bcdec, rgbcx, and bc7enc as vendored external dependencies.

1. **Add bcdec** — Copy `bcdec.h` (single header, MIT/public domain) into `external/include/` alongside other header-only dependencies (`stb_image.h`, etc.). No CMake target needed — already on the include path.
2. **Add bc7enc_rdo files** — Copy `rgbcx.h`, `rgbcx.cpp`, `bc7enc.h`, `bc7enc.cpp` into `external/bc7enc/`. Create a static library CMake target. Disable warnings as needed (follow the `lmdb` pattern in `external/CMakeLists.txt`).
3. **Update `external/CMakeLists.txt`** — Add the bc7enc static library target.
4. **Licenses** — Both bcdec (MIT) and bc7enc_rdo (MIT) are covered by the existing `LICENSES/MIT.txt` template.

**Relevant files:**
- `external/include/bcdec.h` — Header-only BC decoder, included directly.
- `external/CMakeLists.txt` — Follow the `lmdb` pattern: `add_library(bc7enc STATIC)` + source list + warning suppression.
