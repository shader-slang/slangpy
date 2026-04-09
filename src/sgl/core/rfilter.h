// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/error.h"
#include "sgl/core/enum.h"
#include "sgl/core/maths.h"
#include "sgl/math/constants.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <variant>

namespace sgl {

/// Filter boundary condition used for resampling images.
enum class FilterBoundaryCondition {
    /// Clamp to the outermost sample position.
    clamp = 0,
    /// Assume that the input repeats in a periodic fashion.
    repeat,
    /// Assume that the input is mirrored along the boundary.
    mirror,
    /// Assume that the input function is zero outside of the defined domain.
    zero,
    /// Assume that the input function is equal to one outside of the defined domain.
    one,
};
SGL_ENUM_INFO(
    FilterBoundaryCondition,
    {
        {FilterBoundaryCondition::clamp, "clamp"},
        {FilterBoundaryCondition::repeat, "repeat"},
        {FilterBoundaryCondition::mirror, "mirror"},
        {FilterBoundaryCondition::zero, "zero"},
        {FilterBoundaryCondition::one, "one"},
    }
);
SGL_ENUM_REGISTER(FilterBoundaryCondition);

struct BoxFilter {
    BoxFilter() = default;
    float radius() const { return 0.5f; }
    float eval(float x) const { return (x >= -0.5f && x < 0.5f) ? 1.f : 0.f; }
};

struct TentFilter {
    TentFilter(float radius = 1.f)
        : m_radius(radius)
        , m_inv_radius(1.f / radius)
    {
        SGL_CHECK(radius > 0.f, "TentFilter radius must be positive.");
    }
    float radius() const { return m_radius; }
    float eval(float x) const { return std::max(0.f, 1.f - std::abs(x * m_inv_radius)); }

private:
    float m_radius;
    float m_inv_radius;
};

struct GaussianFilter {
    GaussianFilter(float stddev = 0.5f)
        : m_stddev(stddev)
    {
        SGL_CHECK(stddev > 0.f, "GaussianFilter stddev must be positive.");
        m_radius = 4.f * m_stddev;
        m_alpha = -1.f / (2.f * m_stddev * m_stddev);
        m_bias = std::exp(m_alpha * m_radius * m_radius);
    }
    float radius() const { return m_radius; }
    float eval(float x) const
    {
        const float INV_LOG_TWO = 1.44269504088896340736f;
        return std::max(0.f, std::exp2((INV_LOG_TWO * m_alpha) * x * x) - m_bias);
    }

private:
    float m_stddev;
    float m_radius;
    float m_alpha;
    float m_bias;
};

struct MitchellFilter {
    MitchellFilter(float b = 1.f / 3.f, float c = 1.f / 3.f)
        : m_b(b)
        , m_c(c)
    {
    }
    float radius() const { return 2.f; }
    float eval(float x) const
    {
        x = std::abs(x);
        float x2 = x * x;
        float x3 = x2 * x;

        float a3 = (12.f - 9.f * m_b - 6.f * m_c);
        float a2 = (-18.f + 12.f * m_b + 6.f * m_c);
        float a0 = (6.f - 2.f * m_b);
        float b3 = (-m_b - 6.f * m_c);
        float b2 = (6.f * m_b + 30.f * m_c);
        float b1 = (-12.f * m_b - 48.f * m_c);
        float b0 = (8.f * m_b + 24.f * m_c);

        if (x < 1.f)
            return (a3 * x3 + (a2 * x2 + a0)) / 6.f;
        else if (x < 2.f)
            return (b3 * x3 + (b2 * x2 + (b1 * x + b0))) / 6.f;
        else
            return 0.f;
    }

private:
    float m_b;
    float m_c;
};

struct LanczosFilter {
    LanczosFilter(int lobes = 3)
        : m_radius(static_cast<float>(lobes))
    {
        SGL_CHECK(lobes > 0, "LanczosFilter lobes must be positive.");
    }
    float radius() const { return m_radius; }
    float eval(float x) const
    {
        x = std::abs(x);

        float x1 = static_cast<float>(M_PI) * x;
        float x2 = x1 / m_radius;
        if (x < std::numeric_limits<float>::epsilon())
            return 1.f;
        else if (x > m_radius)
            return 0.f;
        else
            return std::sin(x1) * std::sin(x2) / (x1 * x2);
    }

private:
    float m_radius;
};

using ReconstructionFilter = std::variant<BoxFilter, TentFilter, GaussianFilter, MitchellFilter, LanczosFilter>;

/// Utility class for efficiently resampling discrete datasets to different resolutions.
/// \tparam Scalar The underlying floating point data type.
template<typename Scalar_>
struct Resampler {
    using Scalar = Scalar_;
    using Float = float;

    /**
     * \brief Create a new Resampler object that transforms between the specified resolutions.
     *
     * This constructor precomputes all information needed to efficiently perform the
     * desired resampling operation. For that reason, it is most efficient if it can
     * be used repeatedly (e.g. to resample the equal-sized rows of a bitmap)
     *
     * \param source_res Source resolution
     * \param target_res Target resolution
     */
    template<typename Filter>
    Resampler(const Filter& filter, uint32_t source_res, uint32_t target_res)
        : m_source_res(source_res)
        , m_target_res(target_res)
    {
        if (source_res == 0 || target_res == 0)
            SGL_THROW("Resampler::Resampler(): source or target resolution == 0!");

        Float filter_radius_orig = filter.radius();
        Float filter_radius = filter_radius_orig;
        Float scale = Float(1);
        Float inv_scale = Float(1);

        /* Low-pass filter: scale reconstruction filters when downsampling */
        if (target_res < source_res) {
            scale = (Float)source_res / (Float)target_res;
            inv_scale = Float(1) / scale;
            filter_radius *= scale;
        }

        m_taps = static_cast<uint32_t>(std::ceil(filter_radius * 2));
        if (source_res == target_res && (m_taps % 2) != 1)
            --m_taps;

        if (filter_radius_orig < 1)
            m_taps = std::min(m_taps, source_res);

        if (source_res != target_res) { /* Resampling mode */
            m_start = std::unique_ptr<int32_t[]>(new int32_t[target_res]);
            m_weights = std::unique_ptr<Scalar[]>(new Scalar[m_taps * target_res]);
            m_fast_start = 0;
            m_fast_end = m_target_res;

            for (uint32_t i = 0; i < target_res; i++) {
                /* Compute the fractional coordinates of the new sample i
                   in the original coordinates */
                Float center = (i + Float(0.5)) / target_res * source_res;

                /* Determine the index of the first original sample
                   that might contribute */
                m_start[i] = static_cast<int32_t>(std::floor(center - filter_radius + Float(0.5)));

                /* Determine the size of center region, on which to run
                   the fast non condition-aware code */
                if (m_start[i] < 0)
                    m_fast_start = std::max(m_fast_start, i + 1);
                else if (m_start[i] + m_taps - 1 >= m_source_res)
                    m_fast_end = std::min(m_fast_end, i);

                double sum = 0.0;
                for (uint32_t j = 0; j < m_taps; j++) {
                    /* Compute the position where the filter should be evaluated */
                    Float pos = m_start[i] + (int32_t)j + Float(0.5) - center;

                    /* Perform the evaluation and record the weight */
                    auto weight = filter.eval(pos * inv_scale);

                    /* Handle the (numerical) edge case of the pixel center missing
                       the filter support when upsampling using the box filter. */
                    if constexpr (std::is_same_v<std::decay_t<Filter>, BoxFilter>) {
                        if (target_res > source_res)
                            weight = Float(1.0);
                    }
                    m_weights[i * m_taps + j] = static_cast<Scalar>(weight);
                    sum += double(weight);
                }

                SGL_ASSERT(sum != 0);

                /* Normalize the contribution of each sample */
                double normalization = 1.0 / sum;
                for (uint32_t j = 0; j < m_taps; j++) {
                    Scalar& value = m_weights[i * m_taps + j];
                    value = Scalar(double(value) * normalization);
                }
            }
        } else { /* Filtering mode */
            uint32_t half_taps = m_taps / 2;
            m_weights = std::unique_ptr<Scalar[]>(new Scalar[m_taps]);

            double sum = 0.0;
            for (uint32_t i = 0; i < m_taps; i++) {
                auto weight = filter.eval(Float(static_cast<int32_t>(i) - static_cast<int32_t>(half_taps)));
                m_weights[i] = Scalar(weight);
                sum += double(weight);
            }

            SGL_ASSERT(sum != 0);

            double normalization = 1.0 / sum;
            for (uint32_t i = 0; i < m_taps; i++) {
                Scalar& value = m_weights[i];
                value = Scalar(double(value) * normalization);
            }
            m_fast_start = std::min(half_taps, m_target_res - 1);
            m_fast_end = static_cast<uint32_t>(
                std::max(static_cast<int64_t>(m_target_res) - static_cast<int64_t>(half_taps), static_cast<int64_t>(0))
            );
        }

        /* Avoid overlapping fast start/end intervals when the
           target image is very small compared to the source image */
        m_fast_start = std::min(m_fast_start, m_fast_end);
    }

    /// Return the reconstruction filter's source resolution
    uint32_t source_resolution() const { return m_source_res; }

    /// Return the reconstruction filter's target resolution
    uint32_t target_resolution() const { return m_target_res; }

    /// Return the number of taps used by the reconstruction filter
    uint32_t taps() const { return m_taps; }

    /// Boundary condition used when looking up samples outside of the defined input domain.
    FilterBoundaryCondition boundary_condition() const { return m_bc; }

    /// Set the boundary condition used when looking up samples outside of the defined input domain.
    void set_boundary_condition(FilterBoundaryCondition bc) { m_bc = bc; }

    /// Range to which resampled values will be clamped.
    const std::pair<Scalar, Scalar>& clamp() { return m_clamp; }

    /// Set the range to which resampled values will be clamped.
    void set_clamp(const std::pair<Scalar, Scalar>& value) { m_clamp = value; }

    /**
     * \brief Resample a multi-channel array and clamp the results to a specified valid range
     *
     * \param source Source array of samples
     * \param target Target array of samples
     * \param source_stride Stride of samples in the source array. A value of '1' implies that they are densely packed.
     * \param target_stride Stride of samples in the target array. A value of '1' implies that they are densely packed.
     * \param channels Number of channels to be resampled
     */
    void resample(
        const Scalar* source,
        uint32_t source_stride,
        Scalar* target,
        uint32_t target_stride,
        uint32_t channels
    ) const
    {
        using ResampleFunctor = void (Resampler::*)(const Scalar*, uint32_t, Scalar*, uint32_t, uint32_t) const;

        ResampleFunctor f;

        if (m_clamp
            != std::make_pair(-std::numeric_limits<Scalar>::infinity(), std::numeric_limits<Scalar>::infinity())) {
            if (m_start)
                f = &Resampler::resample_internal<true /* Clamp */, true /* Resample */>;
            else
                f = &Resampler::resample_internal<true /* Clamp */, false /* Resample */>;
        } else {
            if (m_start)
                f = &Resampler::resample_internal<false /* Clamp */, true /* Resample */>;
            else
                f = &Resampler::resample_internal<false /* Clamp */, false /* Resample */>;
        }

        (this->*f)(source, source_stride, target, target_stride, channels);
    }

    std::string to_string() const
    {
        return fmt::format("Resampler[source_res={}, target_res={}]", m_source_res, m_target_res);
    }

private:
    template<bool Clamp, bool Resample>
    void resample_internal(
        const Scalar* source,
        uint32_t source_stride,
        Scalar* target,
        uint32_t target_stride,
        uint32_t channels
    ) const
    {
        const uint32_t taps = m_taps, half_taps = m_taps / 2;
        const Scalar* weights = m_weights.get();
        const int32_t* start = m_start.get();
        const Scalar min = std::get<0>(m_clamp);
        const Scalar max = std::get<1>(m_clamp);

        target_stride = channels * (target_stride - 1);
        source_stride *= channels;

        // Resample the left border region, while accounting for the boundary conditions.
        for (uint32_t i = 0; i < m_fast_start; ++i) {
            const int32_t offset = Resample ? (*start++) : (static_cast<int32_t>(i) - half_taps);

            for (uint32_t ch = 0; ch < channels; ++ch) {
                Scalar result = 0;
                for (uint32_t j = 0; j < taps; ++j)
                    result += lookup(source, offset + static_cast<int32_t>(j), source_stride, ch) * weights[j];

                *target++ = Clamp ? std::clamp(result, min, max) : result;
            }

            target += target_stride;

            if (Resample)
                weights += taps;
        }

        // Use a faster branch-free loop for resampling the main portion.
        for (uint32_t i = m_fast_start; i < m_fast_end; ++i) {
            const int32_t offset = Resample ? (*start++) : (static_cast<int32_t>(i) - half_taps);

            for (uint32_t ch = 0; ch < channels; ++ch) {
                Scalar result = 0;
                for (uint32_t j = 0; j < taps; ++j)
                    result += source[source_stride * (offset + static_cast<int32_t>(j)) + ch] * weights[j];

                *target++ = Clamp ? std::clamp(result, min, max) : result;
            }

            target += target_stride;

            if (Resample)
                weights += taps;
        }

        // Resample the right border region, while accounting for the boundary conditions.
        for (uint32_t i = m_fast_end; i < m_target_res; ++i) {
            const int32_t offset = Resample ? (*start++) : (static_cast<int32_t>(i) - half_taps);

            for (uint32_t ch = 0; ch < channels; ++ch) {
                Scalar result = 0;
                for (uint32_t j = 0; j < taps; ++j)
                    result += lookup(source, offset + static_cast<int32_t>(j), source_stride, ch) * weights[j];

                *target++ = Clamp ? std::clamp(result, min, max) : result;
            }

            target += target_stride;

            if (Resample)
                weights += taps;
        }
    }

    Scalar lookup(const Scalar* source, int32_t pos, uint32_t stride, uint32_t ch) const
    {
        if (pos < 0 || pos >= static_cast<int32_t>(m_source_res)) [[unlikely]] {
            switch (m_bc) {
            case FilterBoundaryCondition::clamp:
                pos = std::clamp(pos, 0, static_cast<int32_t>(m_source_res) - 1);
                break;

            case FilterBoundaryCondition::repeat:
                pos = modulo(pos, static_cast<int32_t>(m_source_res));
                break;

            case FilterBoundaryCondition::mirror:
                pos = modulo(pos, 2 * static_cast<int32_t>(m_source_res) - 2);
                if (pos >= static_cast<int32_t>(m_source_res) - 1)
                    pos = 2 * m_source_res - 2 - pos;
                break;

            case FilterBoundaryCondition::one:
                return Scalar(1);

            case FilterBoundaryCondition::zero:
                return Scalar(0);
            }
        }

        return source[pos * stride + ch];
    }

private:
    std::unique_ptr<int32_t[]> m_start;
    std::unique_ptr<Scalar[]> m_weights;
    uint32_t m_source_res;
    uint32_t m_target_res;
    uint32_t m_fast_start;
    uint32_t m_fast_end;
    uint32_t m_taps;
    FilterBoundaryCondition m_bc = FilterBoundaryCondition::clamp;
    std::pair<Scalar, Scalar> m_clamp{
        -std::numeric_limits<Scalar>::infinity(),
        std::numeric_limits<Scalar>::infinity()
    };
};

} // namespace sgl
