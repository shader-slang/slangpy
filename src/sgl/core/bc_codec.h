// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/bc_types.h"
#include "sgl/core/enum.h"

#include <cstddef>
#include <memory>
#include <mutex>

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
SGL_ENUM_INFO(
    BCEncoderBackend,
    {
        {BCEncoderBackend::automatic, "automatic"},
        {BCEncoderBackend::software, "software"},
        {BCEncoderBackend::nvtt_cpu, "nvtt_cpu"},
        {BCEncoderBackend::nvtt_gpu, "nvtt_gpu"},
    }
);
SGL_ENUM_REGISTER(BCEncoderBackend);

/// Reusable BC1-7 block compression encoder. Does not require a Device.
class SGL_API BCEncoder {
public:
    /// Create an encoder using the requested backend.
    /// Throws if the requested backend is unavailable.
    explicit BCEncoder(BCEncoderBackend backend = BCEncoderBackend::automatic);
    ~BCEncoder();

    BCEncoder(const BCEncoder&) = delete;
    BCEncoder& operator=(const BCEncoder&) = delete;

    /// Encode an image to a BC format. If options.generate_mipmaps is true the
    /// full mip chain is generated from the source image.
    BCCompressedImage encode(const BCImage& src, Format format, const BCEncodeOptions& options = {});

    /// Encode a bitmap, converting its color transfer function to match format.
    /// Integer components of all widths are normalized; floating-point values are used directly.
    /// Luminance is expanded to RGB and luminance/alpha to RGBA. R/RG and
    /// multi-channel bitmaps retain channel order. Multi-channel bitmaps are raw
    /// data and must already use the destination transfer function.
    /// The bitmap must be non-null and is borrowed only for the duration of the call.
    /// Calls on a single encoder are serialized. Do not modify inputs during encoding.
    /// Python calls release the GIL during encoding.
    BCCompressedImage encode(const Bitmap* bitmap, Format format, const BCEncodeOptions& options = {});

    /// The resolved backend used by this encoder.
    BCEncoderBackend backend() const { return m_backend; }

    /// True if the given backend is available in this build.
    static bool is_backend_available(BCEncoderBackend backend);

    /// True if the selected backend can encode the given format.
    bool can_encode(Format format) const;

private:
    BCEncoderBackend m_backend;
    std::unique_ptr<BCEncoderImpl> m_impl;
    std::mutex m_mutex;
};

/// Decode a single BC-compressed mip level into the destination image.
SGL_API void
decode_bc(const void* data, size_t size, Format format, uint32_t width, uint32_t height, const BCMutableImage& dst);

} // namespace sgl
