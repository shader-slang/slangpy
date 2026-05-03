// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[AutoPyBindCUDA]
[CUDAKernel]
void sum_slangtorch_1(TensorView<float> tensor_0, TensorView<float> result)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= result.size(0))
        return;
    result[dispatchIdx.x] = tensor_0[dispatchIdx.x];
}

[AutoPyBindCUDA]
[CUDAKernel]
void sum_slangtorch_2(TensorView<float> tensor_0, TensorView<float> tensor_1, TensorView<float> result)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= result.size(0))
        return;
    result[dispatchIdx.x] = tensor_0[dispatchIdx.x] + tensor_1[dispatchIdx.x];
}

[AutoPyBindCUDA]
[CUDAKernel]
void sum_slangtorch_3(TensorView<float> tensor_0, TensorView<float> tensor_1, TensorView<float> tensor_2, TensorView<float> result)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= result.size(0))
        return;
    result[dispatchIdx.x] = tensor_0[dispatchIdx.x] + tensor_1[dispatchIdx.x] + tensor_2[dispatchIdx.x];
}

[AutoPyBindCUDA]
[CUDAKernel]
void sum_slangtorch_4(TensorView<float> tensor_0, TensorView<float> tensor_1, TensorView<float> tensor_2, TensorView<float> tensor_3, TensorView<float> result)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= result.size(0))
        return;
    result[dispatchIdx.x] = tensor_0[dispatchIdx.x] + tensor_1[dispatchIdx.x] + tensor_2[dispatchIdx.x] + tensor_3[dispatchIdx.x];
}

[AutoPyBindCUDA]
[CUDAKernel]
void sum_slangtorch_5(TensorView<float> tensor_0, TensorView<float> tensor_1, TensorView<float> tensor_2, TensorView<float> tensor_3, TensorView<float> tensor_4, TensorView<float> result)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= result.size(0))
        return;
    result[dispatchIdx.x] = tensor_0[dispatchIdx.x] + tensor_1[dispatchIdx.x] + tensor_2[dispatchIdx.x] + tensor_3[dispatchIdx.x] + tensor_4[dispatchIdx.x];
}

[AutoPyBindCUDA]
[CUDAKernel]
void sum_slangtorch_6(TensorView<float> tensor_0, TensorView<float> tensor_1, TensorView<float> tensor_2, TensorView<float> tensor_3, TensorView<float> tensor_4, TensorView<float> tensor_5, TensorView<float> result)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= result.size(0))
        return;
    result[dispatchIdx.x] = tensor_0[dispatchIdx.x] + tensor_1[dispatchIdx.x] + tensor_2[dispatchIdx.x] + tensor_3[dispatchIdx.x] + tensor_4[dispatchIdx.x] + tensor_5[dispatchIdx.x];
}
