// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/bc_types.h"

#include <cstddef>
#include <memory>

namespace sgl {

/// CPU-based BC1–7 block compression encoder/decoder.
///
/// Decoding uses bcdec (header-only). Encoding uses rgbcx (BC1–5) and
/// bc7enc (BC7) for the software backend. BC6H encoding requires NVTT3
/// (loaded at runtime, Phase 5).
class SGL_API BCCodec {
public:
    BCCodec();
    ~BCCodec();

    BCCodec(const BCCodec&) = delete;
    BCCodec& operator=(const BCCodec&) = delete;

    /// Encode an image to a BC format. If options.generate_mipmaps is true the
    /// full mip chain is generated from the source image.
    BCCompressedImage encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options = {});

    /// Decode a single mip level from compressed data into \p dst.
    void decode(
        const void* data,
        size_t size,
        BCFormat format,
        uint32_t width,
        uint32_t height,
        const BCMutableImage& dst
    );

    /// True if the NVTT3 dynamic library was found and loaded.
    bool is_nvtt_available() const;

    /// True if the given format can be encoded (SW or NVTT3).
    bool can_encode(BCFormat format) const;

    /// True if the given format can be decoded (always true for all BC formats).
    bool can_decode(BCFormat format) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace sgl
