// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy;

// Each function computes output[idx] = input[idx] * 3, calling the array/vector
// load/store interface requirements through an interface-typed parameter.

[Differentiable]
void iface_load_array(IDiffTensor<float, 1> input, WDiffTensor<float, 1> output, int idx)
{
    output.store(int[1](idx), input.load(int[1](idx)) * 3.0f);
}

[Differentiable]
void iface_load_vector(IDiffTensor<float, 1> input, WDiffTensor<float, 1> output, int idx)
{
    output.store(int[1](idx), input.load(vector<int, 1>(idx)) * 3.0f);
}

[Differentiable]
void generic_load_impl<T : IDiffTensor<float, 1>>(T input, WDiffTensor<float, 1> output, int idx)
{
    output.store(int[1](idx), input.load(int[1](idx)) * 3.0f);
}

[Differentiable]
void generic_load(DiffTensor<float, 1> input, WDiffTensor<float, 1> output, int idx)
{
    generic_load_impl(input, output, idx);
}

[Differentiable]
void rw_iface_load_array(IRWDiffTensor<float, 1> input, WDiffTensor<float, 1> output, int idx)
{
    output.store(int[1](idx), input.load(int[1](idx)) * 3.0f);
}

[Differentiable]
void rw_iface_load_vector(IRWDiffTensor<float, 1> input, WDiffTensor<float, 1> output, int idx)
{
    output.store(int[1](idx), input.load(vector<int, 1>(idx)) * 3.0f);
}

[Differentiable]
void iface_store_array(DiffTensor<float, 1> input, IWDiffTensor<float, 1> output, int idx)
{
    output.store(int[1](idx), input.load(int[1](idx)) * 3.0f);
}

[Differentiable]
void iface_store_vector(DiffTensor<float, 1> input, IWDiffTensor<float, 1> output, int idx)
{
    output.store(vector<int, 1>(idx), input.load(int[1](idx)) * 3.0f);
}

[Differentiable]
void generic_store_impl<T : IWDiffTensor<float, 1>>(DiffTensor<float, 1> input, T output, int idx)
{
    output.store(int[1](idx), input.load(int[1](idx)) * 3.0f);
}

[Differentiable]
void generic_store(DiffTensor<float, 1> input, WDiffTensor<float, 1> output, int idx)
{
    generic_store_impl(input, output, idx);
}

[Differentiable]
void rw_iface_store_array(DiffTensor<float, 1> input, IRWDiffTensor<float, 1> output, int idx)
{
    output.store(int[1](idx), input.load(int[1](idx)) * 3.0f);
}

[Differentiable]
void rw_iface_store_vector(DiffTensor<float, 1> input, IRWDiffTensor<float, 1> output, int idx)
{
    output.store(vector<int, 1>(idx), input.load(int[1](idx)) * 3.0f);
}
