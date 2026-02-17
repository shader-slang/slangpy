// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple polynomial: a*x^2 + b*x + c
// Forward kernel
[AutoPyBindCUDA]
[CUDAKernel]
void polynomial_fwd(
    float a, float b, float c,
    TensorView<float> x,
    TensorView<float> result)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= result.size(0))
        return;
    float xv = x[dispatchIdx.x];
    result[dispatchIdx.x] = a * xv * xv + b * xv + c;
}

// Backward kernel: d/dx(a*x^2 + b*x + c) = 2*a*x + b
[AutoPyBindCUDA]
[CUDAKernel]
void polynomial_bwd(
    float a, float b, float c,
    TensorView<float> x,
    TensorView<float> grad_output,
    TensorView<float> grad_x)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= x.size(0))
        return;
    float xv = x[dispatchIdx.x];
    grad_x[dispatchIdx.x] = grad_output[dispatchIdx.x] * (2.0 * a * xv + b);
}
