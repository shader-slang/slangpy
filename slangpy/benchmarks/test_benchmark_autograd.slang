// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy;

// Simple polynomial: a*x^2 + b*x + c
// Derivative w.r.t. x: 2*a*x + b
[Differentiable]
float polynomial(float a, float b, float c, float x)
{
    return a * x * x + b * x + c;
}
