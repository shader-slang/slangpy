// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import "slangpy";

[Differentiable]
float polynomial(float a, float b, float c, float x) {
    return a * x * x + b * x + c;
}
