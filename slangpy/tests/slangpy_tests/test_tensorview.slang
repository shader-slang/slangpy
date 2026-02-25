// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

void copy_tensorview(TensorView<float> input, TensorView<float> output)
{
    for (uint i = 0; i < input.size(0); i++)
        output.store(i, input.load(i));
}

void add_tensorview(TensorView<float> a, TensorView<float> b, TensorView<float> output)
{
    for (uint i = 0; i < a.size(0); i++)
        output.store(i, a.load(i) + b.load(i));
}

// Writes a 1 at the given coordinate position.
void mark_thread(int coord, TensorView<int> markers)
{
    markers.store(coord, 1);
}

// Copy float2 elements between TensorView<float2>.
void copy_tensorview_float2(TensorView<float2> input, TensorView<float2> output)
{
    for (uint i = 0; i < input.size(0); i++)
        output.store(i, input.load(i));
}

// Copy float4 elements between TensorView<float4>.
void copy_tensorview_float4(TensorView<float4> input, TensorView<float4> output)
{
    for (uint i = 0; i < input.size(0); i++)
        output.store(i, input.load(i));
}

// CUDAKernel that fills thread IDs - requires _threadcount to dispatch.
[CUDAKernel]
void fill_thread_ids(uint count, TensorView<int> output)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    output.store(tid, int(tid));
}
