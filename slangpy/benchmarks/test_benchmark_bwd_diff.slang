// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy;

// Include loadUniform extension for DiffTensorView
#include "extensions.slang"

// =========================================================================
// Uniform access: y[tid] = w
// All threads load the same weight w and write it to their output slot.
// In backward, gradients for w must be accumulated across all threads —
// this is where contention happens.
// =========================================================================

// --- DiffTensor: load (atomic backward — contention on w) ---
[CUDAKernel]
[Differentiable]
void broadcast_dt(uint count, DiffTensor<float, 1> w, WDiffTensor<float, 1> output)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    float weight = w.load(int[1](0));
    output.store(int[1](tid), weight);
}

// --- DiffTensor: load_uniform (wave-level reduction — correct and fast) ---
[CUDAKernel]
[Differentiable]
void broadcast_dt_uniform(uint count, DiffTensor<float, 1> w, WDiffTensor<float, 1> output)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    float weight = w.load_uniform(int[1](0));
    output.store(int[1](tid), weight);
}

// --- DiffTensorView: load (atomic backward — contention on w) ---
[CUDAKernel]
[Differentiable]
void broadcast_dtv(uint count, DiffTensorView<float> w, DiffTensorView<float> output)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    float weight = w.load(uint(0));
    output.store(tid, weight);
}

// --- DiffTensorView: loadUniform (wave-level reduction — correct and fast) ---
[CUDAKernel]
[Differentiable]
void broadcast_dtv_uniform(uint count, DiffTensorView<float> w, DiffTensorView<float> output)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    float weight = w.loadUniform(uint(0));
    output.store(tid, weight);
}

// =========================================================================
// Per-element access: f(x) = x*x
// Each thread reads/writes a unique index — no contention.
// Non-atomic load/store variants avoid unnecessary atomics.
// =========================================================================

// --- DiffTensor: load (atomic backward, unnecessary for unique indices) ---
[CUDAKernel]
[Differentiable]
void square_dt(uint count, DiffTensor<float, 1> input, WDiffTensor<float, 1> output)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    float x = input.load(int[1](tid));
    output.store(int[1](tid), x * x);
}

// --- DiffTensor: load_once (no atomics, correct for unique indices) ---
[CUDAKernel]
[Differentiable]
void square_dt_once(uint count, DiffTensor<float, 1> input, WDiffTensor<float, 1> output)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    float x = input.load_once(int[1](tid));
    output.store(int[1](tid), x * x);
}

// --- DiffTensorView: load/store (atomic backward, unnecessary) ---
[CUDAKernel]
[Differentiable]
void square_dtv(uint count, DiffTensorView<float> input, DiffTensorView<float> output)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    float x = input.load(tid);
    output.store(tid, x * x);
}

// --- DiffTensorView: loadOnce/storeOnce (no atomics, correct) ---
[CUDAKernel]
[Differentiable]
void square_dtv_once(uint count, DiffTensorView<float> input, DiffTensorView<float> output)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    float x = input.loadOnce(tid);
    output.storeOnce(tid, x * x);
}
