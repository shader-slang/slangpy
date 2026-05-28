// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy;

// Per-element: f(x) = x*x using load_once (no atomics in backward)
[Differentiable]
void square_load_once(DiffTensor<float, 1> input, WDiffTensor<float, 1> output, int idx)
{
    float x = input.load_once(int[1](idx));
    output.store(int[1](idx), x * x);
}

// Per-element: f(x) = x*x using standard load (atomic in backward)
[Differentiable]
void square_load(DiffTensor<float, 1> input, WDiffTensor<float, 1> output, int idx)
{
    float x = input.load(int[1](idx));
    output.store(int[1](idx), x * x);
}

// Uniform: y = w, using load_uniform (wave reduction in backward)
[Differentiable]
void broadcast_load_uniform(DiffTensor<float, 1> w, WDiffTensor<float, 1> output, int idx)
{
    float weight = w.load_uniform(int[1](0));
    output.store(int[1](idx), weight);
}

// Uniform: y = w, using standard load (atomic in backward)
[Differentiable]
void broadcast_load(DiffTensor<float, 1> w, WDiffTensor<float, 1> output, int idx)
{
    float weight = w.load(int[1](0));
    output.store(int[1](idx), weight);
}
