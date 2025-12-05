// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import slangpy;

void copy_itensor_each_index(int[2] idx, ITensor<float, 2> src, IRWTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    dst[i0, i1] = src[i0, i1];
}
void copy_itensor_array_index(int[2] idx, ITensor<float, 2> src, IRWTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    int[2] arr = { i0, i1 };
    dst[arr] = src[arr];
}
void copy_itensor_vector_index(int[2] idx, ITensor<float, 2> src, IRWTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    var vec = int2(i1, i0);
    dst[vec] = src[vec];
}

void copy_idifftensor_each_index(int[2] idx, IDiffTensor<float, 2> src, IRWDiffTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    dst[i0, i1] = src[i0, i1];
}
void copy_idifftensor_array_index(int[2] idx, IDiffTensor<float, 2> src, IRWDiffTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    int[2] arr = { i0, i1 };
    dst[arr] = src[arr];
}
void copy_idifftensor_vector_index(int[2] idx, IDiffTensor<float, 2> src, IRWDiffTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    var vec = int2(i1, i0);
    dst[vec] = src[vec];
}

void copy_tensor_each_index(int[2] idx, Tensor<float, 2> src, RWTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    dst[i0, i1] = src[i0, i1];
}
void copy_tensor_array_index(int[2] idx, Tensor<float, 2> src, RWTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    int[2] arr = {i0, i1};
    dst[arr] = src[arr];

}
void copy_tensor_vector_index(int[2] idx, Tensor<float, 2> src, RWTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    var vec = int2(i1, i0);
    dst[vec] = src[vec];
}
void copy_tensor_each_loadstore(int[2] idx, Tensor<float, 2> src, RWTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    dst.store(i0, i1, src.load(i0, i1));
}

void copy_tensor_array_loadstore(int[2] idx, Tensor<float, 2> src, RWTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    int[2] arr = {i0, i1};
    dst.store(arr, src.load(arr));
}
void copy_tensor_vector_loadstore(int[2] idx, Tensor<float, 2> src, RWTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    var vec = int2(i1, i0);
    dst.store(vec, src.load(vec));
}


void copy_difftensor_each_index(int[2] idx, DiffTensor<float, 2> src, WODiffTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    dst[i0, i1] = src[i0, i1];
}

void copy_difftensor_array_index(int[2] idx, DiffTensor<float, 2> src, WODiffTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    int[2] arr = { i0, i1 };
    dst[arr] = src[arr];
}
void copy_difftensor_vector_index(int[2] idx, DiffTensor<float, 2> src, WODiffTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    var vec = int2(i1, i0);
    dst[vec] = src[vec];
}
void copy_difftensor_each_loadstore(int[2] idx, DiffTensor<float, 2> src, WODiffTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    dst.store(i0, i1, src.load(i0, i1));
}
void copy_difftensor_array_loadstore(int[2] idx, DiffTensor<float, 2> src, WODiffTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    int[2] arr = { i0, i1 };
    dst.store(arr, src.load(arr));
}
void copy_difftensor_vector_loadstore(int[2] idx, DiffTensor<float, 2> src, WODiffTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    var vec = int2(i1, i0);
    dst.store(vec, src.load(vec));
}

void copy_primaltensor_each_index(int[2] idx, PrimalTensor<float, 2> src, WOPrimalTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    dst[i0, i1] = src[i0, i1];
}
void copy_primaltensor_array_index(int[2] idx, PrimalTensor<float, 2> src, WOPrimalTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    int[2] arr = { i0, i1 };
    dst[arr] = src[arr];
}
void copy_primaltensor_vector_index(int[2] idx, PrimalTensor<float, 2> src, WOPrimalTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    var vec = int2(i1, i0);
    dst[vec] = src[vec];
}
void copy_primaltensor_each_loadstore(int[2] idx, PrimalTensor<float, 2> src, WOPrimalTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    dst.store(i0, i1, src.load(i0, i1));
}
void copy_primaltensor_array_loadstore(int[2] idx, PrimalTensor<float, 2> src, WOPrimalTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    int[2] arr = { i0, i1 };
    dst.store(arr, src.load(arr));
}
void copy_primaltensor_vector_loadstore(int[2] idx, PrimalTensor<float, 2> src, WOPrimalTensor<float, 2> dst)
{
    int i0 = idx[0];
    int i1 = idx[1];
    var vec = int2(i1, i0);
    dst.store(vec, src.load(vec));
}


void copy_tensor_by_element(float src, out float dst)
{
    dst = src;
}


[Differentiable]
float[5] matrix_vector_direct(float weights[5][8], float biases[5], float[8] x)
{
    float[5] result;
    for (int r = 0; r < 5; ++r)
    {
        float y = biases[r];
        for (int c = 0; c < 8; ++c)
            y += weights[r][c] * x[c];
        result[r] = y;
    }
    return result;
}

float[5] matrix_vector_tensorized(Tensor<float, 2> weights, Tensor<float, 1> biases, float[8] x)
{
    float[5] result;
    for (int r = 0; r < 5; ++r)
    {
        float y = biases[r];
        for (int c = 0; c < 8; ++c)
            y += weights[r, c] * x[c];

        result[r] = y;
    }
    return result;
}

[Differentiable]
float3 matrix_vector_matrices(float3x4 weights, float4 biases, float[4] x)
{
    float3 result;
    for (int r = 0; r < 3; ++r)
    {
        float y = biases[r];
        for (int c = 0; c < 4; ++c)
            y += weights[r][c] * x[c];

        result[r] = y;
    }
    return result;
}


[Differentiable]
float[5] matrix_vector_interfaces(ITensor<float, 2> weights, ITensor<float, 1> biases, float[8] x)
{
    float[5] result;
    for (int r = 0; r < 5; ++r)
    {
        float y = biases.load(r); // Workaround for slang bug #5921: Have to avoid [] operator, or derivatives vanish
        for (int c = 0; c < 8; ++c)
            y += weights.load(r, c) * x[c];

        result[r] = y;
    }
    return result;
}

[Differentiable]
float[5] matrix_vector_interfaces_diff(IDiffTensor<float, 2> weights, IDiffTensor<float, 1> biases, float[8] x)
{
    float[5] result;
    for (int r = 0; r < 5; ++r)
    {
        float y = biases.load(r); // Workaround for slang bug #5921: Have to avoid [] operator, or derivatives vanish
        for (int c = 0; c < 8; ++c)
            y += weights.load(r, c) * x[c];

        result[r] = y;
    }
    return result;
}


float[FanOut] matrix_vector_generic<let FanIn : int, let FanOut : int>(ITensor<float, 2> weights, ITensor<float, 1> biases, float[FanIn] x)
{
    float[FanOut] result;
    [ForceUnroll]
    for (int r = 0; r < FanOut; ++r)
    {
        float y = biases.load(r);
        [ForceUnroll]
        for (int c = 0; c < FanIn; ++c)
            y += weights.load(r, c) * x[c];

        result[r] = y;
    }
    return result;
}

[Differentiable]
float[FanOut] matrix_vector_generic_diff<let FanIn : int, let FanOut : int>( IDiffTensor<float, 2> weights, IDiffTensor<float, 1> biases, float[FanIn] x )
{
    float[FanOut] result;
    [ForceUnroll]
    for (int r = 0; r < FanOut; ++r)
    {
        float y = biases.load(r);
        [ForceUnroll]
        for (int c = 0; c < FanIn; ++c)
            y += weights.load(r, c) * x[c];

        result[r] = y;
    }
    return result;
}
