# Phase 5: NVTT3 Dynamic Loading

Optional accelerated backend for all formats including BC6H encoding. Added after SW path is working end-to-end.

## 5a. NVTT3 Dynamic Loading (`bc_codec.cpp`, private implementation)

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

## 5b. NVTT3 Tests

- **NVTT3 backend** (conditional) — If NVTT3 is available, test BC6H encoding, test that NVTT3 backend produces valid output for all formats
- Add conditional C++ test cases in `test_bc_codec.cpp` that check `is_nvtt_available()` and skip if unavailable
