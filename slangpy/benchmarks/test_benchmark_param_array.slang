// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy;

// Benchmarks for the cost model of Slang's CUDA dynamic-index legalization
// (shader-slang/slang#11939): a runtime index into a by-value kernel parameter
// array is rewritten to go through an eager, whole-aggregate per-thread local
// copy made in the kernel prologue. These kernels sweep the two axes that
// bound that trade:
//
//   - N: the parameter array size (the size of the eager copy)
//   - accesses per thread: 1 (worst case for the copy: pay N, use 1)
//     vs N (best case: copy amortized over many O(1) reads)
//
// plus a statically-indexed control where the legalization must not fire at
// all, isolating the copy cost from everything else.
//
// The array is plain data (float[N]) so that on CUDA it stays a by-value
// entry-point argument under both the old and new ABI (the by-reference
// conversion only applies to descriptor-bearing structs); the same script run
// against a pre-#11939 compiler therefore measures the serial `.param` chain
// this pass replaces.

// One dynamic access per thread: the eager copy of all N elements is paid to
// read a single element. This is the shape most likely to regress as N grows.
float pick_one<let N : int>(uint2 tid, float weights[N], Tensor<uint, 2> indices)
{
    return weights[indices.load(tid)];
}

// N dynamic accesses per thread: the copy is amortized over N O(1) reads.
// This is the shape the legalization exists to accelerate.
float sum_dynamic<let N : int>(uint2 tid, float weights[N], Tensor<uint, 2> indices)
{
    float result = 0.0;
    uint base = indices.load(tid);
    for (int i = 0; i < N; i++)
    {
        result += weights[(base + i) % N];
    }
    return result;
}

// Control: every index is a compile-time constant after unrolling, so the
// legalization must not fire and no local copy is made. Any time delta between
// this and sum_dynamic bounds the total cost of copy + dynamic addressing.
float sum_static<let N : int>(uint2 tid, float weights[N])
{
    float result = 0.0;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        result += weights[i];
    }
    return result;
}
