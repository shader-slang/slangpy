// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "downsample.h"

#include "sgl/core/error.h"
#include "sgl/math/colorspace.h"

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <type_traits>

#if SGL_X86_64
#if SGL_MSVC
#include <intrin.h>
#define SGL_SSSE3_TARGET
#else
#include <cpuid.h>
#define SGL_SSSE3_TARGET __attribute__((target("ssse3")))
#endif
#include <immintrin.h>
#elif SGL_ARM64
#include <arm_neon.h>
#endif

namespace sgl {

namespace {

    /// Lookup tables between 8-bit sRGB and 16-bit linear values.
    struct SrgbTables {
        std::array<uint16_t, 256> decode;
        std::array<uint8_t, 65536> encode;

        static const SrgbTables& get()
        {
            static const std::unique_ptr<SrgbTables> tables = build();
            return *tables;
        }

        /// Average four sRGB samples in linear light.
        uint8_t average(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const
        {
            return encode[(uint32_t(decode[a]) + decode[b] + decode[c] + decode[d] + 2) >> 2];
        }

    private:
        static std::unique_ptr<SrgbTables> build()
        {
            auto tables = std::make_unique<SrgbTables>();
            for (uint32_t i = 0; i < 256; ++i)
                tables->decode[i]
                    = static_cast<uint16_t>(std::lround(math::srgb_to_linear(double(i) / 255.0) * 65535.0));
            for (uint32_t i = 0; i < 65536; ++i)
                tables->encode[i]
                    = static_cast<uint8_t>(std::lround(math::linear_to_srgb(double(i) / 65535.0) * 255.0));
            return tables;
        }
    };

    inline uint8_t average_linear(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
    {
        return static_cast<uint8_t>((a + b + c + d + 2) >> 2);
    }

    // ------------------------------------------------------------------------
    // float32 and unorm16
    // ------------------------------------------------------------------------

    /// Average four samples `a`, `b` (row 0) and `c`, `d` (row 1).
    inline float average4(float a, float b, float c, float d)
    {
        // Same operation order as the separable resampler: horizontal pass, then vertical pass.
        const float h0 = a * 0.5f + b * 0.5f;
        const float h1 = c * 0.5f + d * 0.5f;
        return h0 * 0.5f + h1 * 0.5f;
    }

    inline uint16_t average4(uint16_t a, uint16_t b, uint16_t c, uint16_t d)
    {
        return static_cast<uint16_t>((uint32_t(a) + b + c + d + 2) >> 2);
    }

    /// Scalar kernel for element types without transfer function. `C == 0` handles any channel count.
    template<typename T, uint32_t C>
    void downsample_plain(
        const T* src,
        size_t src_row_pitch,
        T* dst,
        size_t dst_row_pitch,
        uint32_t dst_width,
        uint32_t dst_height,
        uint32_t channel_count
    )
    {
        const uint32_t channels = C != 0 ? C : channel_count;
        for (uint32_t y = 0; y < dst_height; ++y) {
            const auto* row0
                = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(src) + 2 * y * src_row_pitch);
            const auto* row1 = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(row0) + src_row_pitch);
            auto* out = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(dst) + y * dst_row_pitch);
            for (uint32_t x = 0; x < dst_width; ++x) {
                for (uint32_t c = 0; c < channels; ++c)
                    out[c] = average4(row0[c], row0[channels + c], row1[c], row1[channels + c]);
                row0 += 2 * channels;
                row1 += 2 * channels;
                out += channels;
            }
        }
    }

    template<typename T>
    void dispatch_plain(
        const T* src,
        size_t src_row_pitch,
        T* dst,
        size_t dst_row_pitch,
        uint32_t dst_width,
        uint32_t dst_height,
        uint32_t channel_count
    )
    {
        SGL_CHECK(channel_count > 0, "Channel count must be positive.");
        auto run = [&](auto kernel)
        {
            kernel(src, src_row_pitch, dst, dst_row_pitch, dst_width, dst_height, channel_count);
        };
        switch (channel_count) {
        case 1:
            return run(downsample_plain<T, 1>);
        case 2:
            return run(downsample_plain<T, 2>);
        case 3:
            return run(downsample_plain<T, 3>);
        case 4:
            return run(downsample_plain<T, 4>);
        default:
            return run(downsample_plain<T, 0>);
        }
    }

    // ------------------------------------------------------------------------
    // unorm8 row kernels
    //
    // Each row kernel reduces two source rows into one destination row. SIMD kernels process a prefix of the row and
    // return the number of destination pixels written; the scalar kernel finishes the rest.
    // ------------------------------------------------------------------------

    /// Scalar row kernel for 1-4 channels. With `Srgb`, all channels but the last one (if `HasAlpha`) are averaged
    /// in linear light.
    template<uint32_t C, bool Srgb, bool HasAlpha>
    void unorm8_row_scalar(
        const uint8_t* row0,
        const uint8_t* row1,
        uint8_t* out,
        uint32_t x_begin,
        uint32_t x_end,
        const SrgbTables* tables
    )
    {
        constexpr uint32_t color_channels = HasAlpha ? C - 1 : C;
        row0 += size_t(2) * x_begin * C;
        row1 += size_t(2) * x_begin * C;
        out += size_t(x_begin) * C;
        for (uint32_t x = x_begin; x < x_end; ++x) {
            for (uint32_t c = 0; c < color_channels; ++c) {
                if constexpr (Srgb)
                    out[c] = tables->average(row0[c], row0[C + c], row1[c], row1[C + c]);
                else
                    out[c] = average_linear(row0[c], row0[C + c], row1[c], row1[C + c]);
            }
            if constexpr (HasAlpha)
                out[C - 1] = average_linear(row0[C - 1], row0[2 * C - 1], row1[C - 1], row1[2 * C - 1]);
            row0 += 2 * C;
            row1 += 2 * C;
            out += C;
        }
    }

#if SGL_X86_64
    bool cpu_has_ssse3()
    {
#if SGL_MSVC
        int info[4] = {0};
        __cpuid(info, 1);
        return (info[2] & (1 << 9)) != 0; // ECX bit 9 = SSSE3
#else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx))
            return (ecx & (1 << 9)) != 0;
        return false;
#endif
    }

    /// Byte shuffle that moves the two source pixels of each destination pixel next to each other, channel by
    /// channel, so that `pmaddubsw` sums each horizontal pair.
    template<uint32_t C>
    SGL_SSSE3_TARGET __m128i pair_shuffle_mask()
    {
        if constexpr (C == 2)
            return _mm_setr_epi8(0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15);
        else if constexpr (C == 3)
            return _mm_setr_epi8(0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11, -1, -1, -1, -1);
        else if constexpr (C == 4)
            return _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
        else
            return _mm_setzero_si128(); // C == 1: pairs are already adjacent.
    }

    /// Sum each horizontal pair of a 16-byte source chunk of both rows into 16-bit lanes.
    template<uint32_t C>
    SGL_SSSE3_TARGET __m128i box_sum_ssse3(const uint8_t* p0, const uint8_t* p1, __m128i shuffle, __m128i ones)
    {
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p0));
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1));
        if constexpr (C != 1) {
            a = _mm_shuffle_epi8(a, shuffle);
            b = _mm_shuffle_epi8(b, shuffle);
        }
        return _mm_add_epi16(_mm_maddubs_epi16(a, ones), _mm_maddubs_epi16(b, ones));
    }

    /// Linear-light SSSE3 row kernel for 1-4 channels, computing exactly `(a + b + c + d + 2) >> 2`.
    template<uint32_t C>
    SGL_SSSE3_TARGET uint32_t
    unorm8_row_linear_ssse3(const uint8_t* row0, const uint8_t* row1, uint8_t* out, uint32_t width)
    {
        const __m128i shuffle = pair_shuffle_mask<C>();
        const __m128i ones = _mm_set1_epi8(1);
        const __m128i two = _mm_set1_epi16(2);
        uint32_t x = 0;
        if constexpr (C == 3) {
            // 4 destination pixels per iteration from two 12-byte chunks per row. The second 16-byte load reads
            // 4 bytes past the chunk, so stop while those bytes are still inside the row.
            const __m128i compact = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, -1, -1, -1, -1);
            for (; size_t(x + 4) * 6 + 4 <= size_t(width) * 6; x += 4) {
                const uint8_t* p0 = row0 + size_t(x) * 6;
                const uint8_t* p1 = row1 + size_t(x) * 6;
                __m128i s0 = box_sum_ssse3<3>(p0, p1, shuffle, ones);
                __m128i s1 = box_sum_ssse3<3>(p0 + 12, p1 + 12, shuffle, ones);
                s0 = _mm_srli_epi16(_mm_add_epi16(s0, two), 2);
                s1 = _mm_srli_epi16(_mm_add_epi16(s1, two), 2);
                const __m128i result = _mm_shuffle_epi8(_mm_packus_epi16(s0, s1), compact);
                uint8_t* o = out + size_t(x) * 3;
                _mm_storel_epi64(reinterpret_cast<__m128i*>(o), result);
                const int32_t tail = _mm_cvtsi128_si32(_mm_srli_si128(result, 8));
                std::memcpy(o + 8, &tail, 4);
            }
        } else {
            // 16 destination bytes per iteration from 32 source bytes per row.
            constexpr uint32_t step = 16 / C;
            for (; x + step <= width; x += step) {
                const uint8_t* p0 = row0 + size_t(x) * 2 * C;
                const uint8_t* p1 = row1 + size_t(x) * 2 * C;
                __m128i s0 = box_sum_ssse3<C>(p0, p1, shuffle, ones);
                __m128i s1 = box_sum_ssse3<C>(p0 + 16, p1 + 16, shuffle, ones);
                s0 = _mm_srli_epi16(_mm_add_epi16(s0, two), 2);
                s1 = _mm_srli_epi16(_mm_add_epi16(s1, two), 2);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out + size_t(x) * C), _mm_packus_epi16(s0, s1));
            }
        }
        return x;
    }
#endif // SGL_X86_64

#if SGL_ARM64
    /// Split 32 bytes into even and odd pixels of `C` bytes each.
    template<uint32_t C>
    inline void load_even_odd(const uint8_t* p, uint8x16_t& even, uint8x16_t& odd)
    {
        const uint8x16_t lo = vld1q_u8(p);
        const uint8x16_t hi = vld1q_u8(p + 16);
        if constexpr (C == 1) {
            even = vuzp1q_u8(lo, hi);
            odd = vuzp2q_u8(lo, hi);
        } else if constexpr (C == 2) {
            even = vreinterpretq_u8_u16(vuzp1q_u16(vreinterpretq_u16_u8(lo), vreinterpretq_u16_u8(hi)));
            odd = vreinterpretq_u8_u16(vuzp2q_u16(vreinterpretq_u16_u8(lo), vreinterpretq_u16_u8(hi)));
        } else {
            static_assert(C == 4);
            even = vreinterpretq_u8_u32(vuzp1q_u32(vreinterpretq_u32_u8(lo), vreinterpretq_u32_u8(hi)));
            odd = vreinterpretq_u8_u32(vuzp2q_u32(vreinterpretq_u32_u8(lo), vreinterpretq_u32_u8(hi)));
        }
    }

    /// Linear-light NEON row kernel for 1-4 channels, computing exactly `(a + b + c + d + 2) >> 2`.
    template<uint32_t C>
    uint32_t unorm8_row_linear_neon(const uint8_t* row0, const uint8_t* row1, uint8_t* out, uint32_t width)
    {
        uint32_t x = 0;
        if constexpr (C == 3) {
            // De-interleave 16 source pixels per row into channel planes, sum adjacent pairs and re-interleave.
            for (; x + 8 <= width; x += 8) {
                const uint8x16x3_t a = vld3q_u8(row0 + size_t(x) * 6);
                const uint8x16x3_t b = vld3q_u8(row1 + size_t(x) * 6);
                uint8x8x3_t result;
                for (int c = 0; c < 3; ++c)
                    result.val[c] = vrshrn_n_u16(vpadalq_u8(vpaddlq_u8(a.val[c]), b.val[c]), 2);
                vst3_u8(out + size_t(x) * 3, result);
            }
        } else {
            // 16 destination bytes per iteration: add even and odd source pixels of both rows lane by lane.
            constexpr uint32_t step = 16 / C;
            for (; x + step <= width; x += step) {
                uint8x16_t a_even, a_odd, b_even, b_odd;
                load_even_odd<C>(row0 + size_t(x) * 2 * C, a_even, a_odd);
                load_even_odd<C>(row1 + size_t(x) * 2 * C, b_even, b_odd);
                const uint16x8_t lo = vaddq_u16(
                    vaddl_u8(vget_low_u8(a_even), vget_low_u8(a_odd)),
                    vaddl_u8(vget_low_u8(b_even), vget_low_u8(b_odd))
                );
                const uint16x8_t hi = vaddq_u16(vaddl_high_u8(a_even, a_odd), vaddl_high_u8(b_even, b_odd));
                vst1q_u8(out + size_t(x) * C, vcombine_u8(vrshrn_n_u16(lo, 2), vrshrn_n_u16(hi, 2)));
            }
        }
        return x;
    }
#endif // SGL_ARM64

    using LinearRowKernel = uint32_t (*)(const uint8_t*, const uint8_t*, uint8_t*, uint32_t);

    /// SIMD linear-light row kernel for this CPU, or nullptr.
    template<uint32_t C>
    LinearRowKernel linear_row_kernel_simd()
    {
#if SGL_X86_64
        static const bool has_ssse3 = cpu_has_ssse3();
        return has_ssse3 ? &unorm8_row_linear_ssse3<C> : nullptr;
#elif SGL_ARM64
        return &unorm8_row_linear_neon<C>;
#else
        return nullptr;
#endif
    }

    template<uint32_t C, bool Srgb, bool HasAlpha>
    void downsample_unorm8(
        const uint8_t* src,
        size_t src_row_pitch,
        uint8_t* dst,
        size_t dst_row_pitch,
        uint32_t dst_width,
        uint32_t dst_height,
        bool use_simd
    )
    {
        const SrgbTables* tables = Srgb ? &SrgbTables::get() : nullptr;
        // Alpha is linear like everything else when not decoding sRGB, so the linear kernel covers it.
        const LinearRowKernel simd = !Srgb && use_simd ? linear_row_kernel_simd<C>() : nullptr;
        for (uint32_t y = 0; y < dst_height; ++y) {
            const uint8_t* row0 = src + 2 * y * src_row_pitch;
            const uint8_t* row1 = row0 + src_row_pitch;
            uint8_t* out = dst + y * dst_row_pitch;
            const uint32_t x = simd ? simd(row0, row1, out, dst_width) : 0;
            unorm8_row_scalar<C, Srgb, HasAlpha>(row0, row1, out, x, dst_width, tables);
        }
    }

    /// Generic scalar kernel for any channel count. With `srgb`, all channels but the last one (if `has_alpha`) are
    /// averaged in linear light.
    void downsample_unorm8_generic(
        const uint8_t* src,
        size_t src_row_pitch,
        uint8_t* dst,
        size_t dst_row_pitch,
        uint32_t dst_width,
        uint32_t dst_height,
        uint32_t channels,
        bool srgb,
        bool has_alpha
    )
    {
        const SrgbTables* tables = srgb ? &SrgbTables::get() : nullptr;
        const uint32_t srgb_channels = srgb ? (has_alpha ? channels - 1 : channels) : 0;
        for (uint32_t y = 0; y < dst_height; ++y) {
            const uint8_t* row0 = src + 2 * y * src_row_pitch;
            const uint8_t* row1 = row0 + src_row_pitch;
            uint8_t* out = dst + y * dst_row_pitch;
            for (uint32_t x = 0; x < dst_width; ++x) {
                for (uint32_t c = 0; c < channels; ++c) {
                    const uint32_t a = row0[c], b = row0[channels + c], d = row1[c], e = row1[channels + c];
                    out[c] = c < srgb_channels ? tables->average(a, b, d, e) : average_linear(a, b, d, e);
                }
                row0 += 2 * channels;
                row1 += 2 * channels;
                out += channels;
            }
        }
    }

    void downsample_unorm8_dispatch(
        const uint8_t* src,
        size_t src_row_pitch,
        uint8_t* dst,
        size_t dst_row_pitch,
        uint32_t dst_width,
        uint32_t dst_height,
        uint32_t channel_count,
        bool srgb,
        bool has_alpha,
        bool use_simd
    )
    {
        SGL_CHECK(channel_count > 0, "Channel count must be positive.");

        if (channel_count > 4) {
            downsample_unorm8_generic(
                src,
                src_row_pitch,
                dst,
                dst_row_pitch,
                dst_width,
                dst_height,
                channel_count,
                srgb,
                has_alpha
            );
            return;
        }

        // Without sRGB, or for a lone alpha channel, all channels are linear.
        const bool linear = !srgb || (has_alpha && channel_count == 1);
        auto run = [&](auto channels)
        {
            constexpr uint32_t C = decltype(channels)::value;
            auto kernel = linear ? &downsample_unorm8<C, false, false>
                : has_alpha      ? &downsample_unorm8<C, true, true>
                                 : &downsample_unorm8<C, true, false>;
            kernel(src, src_row_pitch, dst, dst_row_pitch, dst_width, dst_height, use_simd);
        };
        switch (channel_count) {
        case 1:
            return run(std::integral_constant<uint32_t, 1>{});
        case 2:
            return run(std::integral_constant<uint32_t, 2>{});
        case 3:
            return run(std::integral_constant<uint32_t, 3>{});
        case 4:
            return run(std::integral_constant<uint32_t, 4>{});
        }
    }

} // namespace

void downsample_2x_box(
    const float* src,
    size_t src_row_pitch,
    float* dst,
    size_t dst_row_pitch,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t channel_count
)
{
    dispatch_plain(src, src_row_pitch, dst, dst_row_pitch, dst_width, dst_height, channel_count);
}

void downsample_2x_box_unorm16(
    const uint16_t* src,
    size_t src_row_pitch,
    uint16_t* dst,
    size_t dst_row_pitch,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t channel_count
)
{
    dispatch_plain(src, src_row_pitch, dst, dst_row_pitch, dst_width, dst_height, channel_count);
}

void downsample_2x_box_unorm8(
    const uint8_t* src,
    size_t src_row_pitch,
    uint8_t* dst,
    size_t dst_row_pitch,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t channel_count,
    bool srgb,
    bool has_alpha
)
{
    downsample_unorm8_dispatch(
        src,
        src_row_pitch,
        dst,
        dst_row_pitch,
        dst_width,
        dst_height,
        channel_count,
        srgb,
        has_alpha,
        true
    );
}

namespace detail {
    void downsample_2x_box_unorm8_scalar(
        const uint8_t* src,
        size_t src_row_pitch,
        uint8_t* dst,
        size_t dst_row_pitch,
        uint32_t dst_width,
        uint32_t dst_height,
        uint32_t channel_count,
        bool srgb,
        bool has_alpha
    )
    {
        downsample_unorm8_dispatch(
            src,
            src_row_pitch,
            dst,
            dst_row_pitch,
            dst_width,
            dst_height,
            channel_count,
            srgb,
            has_alpha,
            false
        );
    }
} // namespace detail

} // namespace sgl
