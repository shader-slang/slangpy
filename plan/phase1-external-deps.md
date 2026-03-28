# Phase 1: External Dependencies

Add bcdec, rgbcx, and bc7enc as vendored external dependencies.

1. **Add bcdec** — Copy `bcdec.h` (single header, MIT/public domain) into `external/bcdec/`. Create a minimal CMake target (interface library, header-only). Add license file.
2. **Add bc7enc_rdo files** — Copy `rgbcx.h`, `rgbcx.cpp`, `bc7enc.h`, `bc7enc.cpp` into `external/bc7enc/`. Create a static library CMake target. Disable warnings as needed (follow the `lmdb` pattern in `external/CMakeLists.txt`). Add license file.
3. **Update `external/CMakeLists.txt`** — Add the two new subdirectories/targets.
4. **Update `LICENSES/`** — Add license texts for bcdec (MIT) and bc7enc_rdo (MIT). Verify exact license terms from upstream repos before vendoring.

**Relevant files:**
- `external/CMakeLists.txt` — Follow the `lmdb` pattern: `add_library(bcdec INTERFACE)` + `target_include_directories(bcdec INTERFACE bcdec)`, and `add_library(bc7enc STATIC)` + source list + warning suppression.
