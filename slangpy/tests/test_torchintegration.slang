// SPDX-License-Identifier: Apache-2.0

[Differentiable]
float add(float a, float b) {
    return a + b;
}

[Differentiable]
float3 add_vectors(float3 a, float3 b) {
    return a + b;
}

[Differentiable]
vector<float,N> add_vectors_generic<let N:int>(vector<float,N> a, vector<float,N> b) {
    return a + b;
}

[Differentiable]
float[5] add_arrays(float[5] a, float[5] b) {
    float[5] res;
    for (int i = 0; i < 5; i++)
        res[i] = a[i] + b[i];
    return res;
}


[Differentiable]
float polynomial(float a, float b, float c, float x) {
    return a * x * x + b * x + c;
}

[Differentiable]
void polynomial_outparam(float a, float b, float c, float x, out float res) {
    res = a * x * x + b * x + c;
}

