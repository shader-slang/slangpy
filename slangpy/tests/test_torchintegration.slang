// SPDX-License-Identifier: Apache-2.0
import slangpy;

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
void add_tensors(Tensor<float, 2> a, Tensor<float, 2> b, RWTensor<float, 2> res) {
    [MaxIters(8)]
    for (int i = 0; i < a.shape[0]; i++)
        [MaxIters(8)]
        for (int j = 0; j < a.shape[1]; j++)
            res.set({i, j}, a[i,j] + b[i, j]);
}

[Differentiable]
float[5] add_arrays(float[5] a, float[5] b) {
    float[5] res;
    for (int i = 0; i < 5; i++)
        res[i] = a[i] + b[i];
    return res;
}

[Differentiable]
void add_out(float a, float b, out float res) {
    res = a + b;
}

[Differentiable]
void add_vectors_out(float3 a, float3 b, out float3 res) {
    res = a + b;
}

[Differentiable]
void add_vectors_generic_out<let N : int>(vector<float, N> a, vector<float, N> b, out vector<float, N> res) {
    res = a + b;
}

[Differentiable]
void add_arrays_out(float[5] a, float[5] b, out float[5] res) {
    for (int i = 0; i < 5; i++)
        res[i] = a[i] + b[i];
}

[Differentiable]
float polynomial(float a, float b, float c, float x) {
    return a * x * x + b * x + c;
}

[Differentiable]
void polynomial_out(float a, float b, float c, float x, out float res) {
    res = a * x * x + b * x + c;
}

[Differentiable]
float3 polynomial_vectors(float a, float b, float c, float3 x) {
    return a * x * x + b * x + c;
}

[Differentiable]
void polynomial_vectors_out(float a, float b, float c, float3 x, out float3 res) {
    res = a * x * x + b * x + c;
}

[Differentiable]
float[5] polynomial_arrays(float a, float b, float c, float[5] x) {
    float[5] res;
    for (int i = 0; i < 5; i++)
        res[i] = a * x[i] * x[i] + b * x[i] + c;
    return res;
}

[Differentiable]
void polynomial_arrays_out(float a, float b, float c, float[5] x, out float[5] res) {
    for (int i = 0; i < 5; i++)
        res[i] = a * x[i] * x[i] + b * x[i] + c;
}