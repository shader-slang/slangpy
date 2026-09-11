// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/bc_types.h"

#include <cstddef>
#include <memory>

namespace sgl {

struct BCEncoderImpl;

/// Encoding backend used by BCEncoder.
enum class BCEncoderBackend {
    /// Prefer the NVTT GPU encoder, then NVTT CPU, then the software encoder.
    automatic,
    /// Use the built-in rgbcx/bc7enc software encoder.
    software,
    /// Require the optional NVTT encoder and run it on the CPU.
    nvtt_cpu,
    /// Require the optional NVTT encoder and CUDA support.
    nvtt_gpu,
};

/// BC1-7 block compression encoder.
class SGL_API BCEncoder {
public:
    explicit BCEncoder(BCEncoderBackend backend = BCEncoderBackend::automatic);
    ~BCEncoder();

    BCEncoder(const BCEncoder&) = delete;
    BCEncoder& operator=(const BCEncoder&) = delete;

    /// Encode an image to a BC format. If options.generate_mipmaps is true the
    /// full mip chain is generated from the source image.
    BCCompressedImage encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options = {});

    /// The resolved backend used by this encoder.
    BCEncoderBackend backend() const { return m_backend; }

    /// True if the given backend is available in this build.
    static bool is_backend_available(BCEncoderBackend backend);

    /// True if the selected backend can encode the given format.
    bool can_encode(BCFormat format) const;

private:
    BCEncoderBackend m_backend;
    std::unique_ptr<BCEncoderImpl> m_impl;
};

/// Decode a single BC-compressed mip level into the destination image.
SGL_API void
decode_bc(const void* data, size_t size, BCFormat format, uint32_t width, uint32_t height, const BCMutableImage& dst);

} // namespace sgl
