// SPDX-License-Identifier: Apache-2.0
import "slangpy";

[Differentiable]
float polynomial(float a, float b, float c, float x) {
    return a * x * x + b * x + c;
}
