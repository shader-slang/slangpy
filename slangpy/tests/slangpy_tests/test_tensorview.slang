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
