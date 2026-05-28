// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "macros.h"

#include <cstdint>

namespace sgl {

/// High resolution CPU timer.
class SGL_API Timer {
public:
    // Time point in nanoseconds.
    using TimePoint = uint64_t;

    Timer() { reset(); }

    /// Reset the timer.
    void reset() { m_start = now(); }

    /// Elapsed seconds since last reset.
    double elapsed_s() const { return delta_s(m_start, now()); }

    /// Elapsed milliseconds since last reset.
    double elapsed_ms() const { return delta_ms(m_start, now()); }

    /// Elapsed microseconds since last reset.
    double elapsed_us() const { return delta_us(m_start, now()); }

    /// Elapsed nanoseconds since last reset.
    double elapsed_ns() const { return delta_ns(m_start, now()); }

    /// Compute elapsed seconds between two time points.
    static double delta_s(TimePoint start, TimePoint end) { return (end - start) * 1e-9; }

    /// Compute elapsed milliseconds between two time points.
    static double delta_ms(TimePoint start, TimePoint end) { return (end - start) * 1e-6; }

    /// Compute elapsed microseconds between two time points.
    static double delta_us(TimePoint start, TimePoint end) { return (end - start) * 1e-3; }

    /// Compute elapsed nanoseconds between two time points.
    static double delta_ns(TimePoint start, TimePoint end) { return double(end - start); }

    /// Current time point in nanoseconds since epoch.
    static TimePoint now();

private:
    TimePoint m_start;
};
} // namespace sgl
