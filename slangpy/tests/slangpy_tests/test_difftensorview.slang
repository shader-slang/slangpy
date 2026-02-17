// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

void copy_difftensorview(DiffTensorView<float> input, DiffTensorView<float> output)
{
    for (uint i = 0; i < input.size(0); i++)
        output.store(i, input.load(i));
}

void add_difftensorview(DiffTensorView<float> a, DiffTensorView<float> b, DiffTensorView<float> output)
{
    for (uint i = 0; i < a.size(0); i++)
        output.store(i, a.load(i) + b.load(i));
}

[Differentiable]
void diff_square(DiffTensorView<float> input, DiffTensorView<float> output)
{
    [MaxIters(1024)]
    for (uint i = 0; i < input.size(0); i++)
    {
        float x = input.load(i);
        output.store(i, x * x);
    }
}
