// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import coverage_color;

// Huber loss fits middle gray while limiting the influence of bright outliers.
[Differentiable]
float exposureLoss(float exposure, no_diff float light)
{
    float error = exposure * light - 0.45;
    if (error < -0.25)
        return -0.25 * error - 0.03125;
    if (error > 0.25)
        return 0.25 * error - 0.03125;
    return 0.5 * error * error;
}

float3 adjustExposure(float3 color, float exposure)
{
    float light = luminance(color);
    if (light <= 1e-6)
        return color;
    // Stop on convergence, on the lower exposure bound, or after four steps.
    for (int iteration = 0; iteration < 4; ++iteration)
    {
        let loss = fwd_diff(exposureLoss)(diffPair(exposure, 1.0), light);
        if (abs(loss.d) < 0.001)
            break;
        float nextExposure = exposure - 0.25 * loss.d;
        if (nextExposure < 0.25)
        {
            exposure = 0.25;
            break;
        }
        exposure = nextExposure;
    }
    return color * exposure;
}
