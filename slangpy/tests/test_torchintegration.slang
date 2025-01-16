// SPDX-License-Identifier: Apache-2.0


float add(float a, float b) {
    return a + b;
}

[Differentiable]
float polynomial(float a, float b, float c, float x) {
    return a * x * x + b * x + c;
}

[Differentiable]
void polynomial_outparam(float a, float b, float c, float x, out float res) {
    res = a * x * x + b * x + c;
}

