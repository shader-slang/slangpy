// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy;

float add(float a, float b)
{
    return a + b;
}

float sum<let N : int>(uint2 tid, RWTensor<float, 2> tensors[N])
{
    float result = 0.0;
    for (int i = 0; i < N; i++)
    {
        result += tensors[i].load(tid);
    }
    return result;
}

struct TensorList<let N : int>
{
    RWTensor<float, 2> tensors[N];

    float fetch(int tensor_index, uint2 index)
    {
        return tensors[tensor_index].load(index);
    }
}

float sum_indirect<let N : int>(uint2 tid, TensorList<N> tensor_list, uint tensor_indices[N])
{
    float result = 0.0;
    for (int i = 0; i < N; i++)
    {
        result += tensor_list.fetch(tensor_indices[i], tid);
    }
    return result;
}

float sum(Tensor<float, 2> tensor_0)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0,0));
    return result;
}

float sum(Tensor<float, 2> tensor_0, Tensor<float, 2> tensor_1)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0,0));
    result += tensor_1.load(uint2(0,0));
    return result;
}

float sum(Tensor<float, 2> tensor_0, Tensor<float, 2> tensor_1, Tensor<float, 2> tensor_2)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0,0));
    result += tensor_1.load(uint2(0,0));
    result += tensor_2.load(uint2(0,0));
    return result;
}

float sum(Tensor<float, 2> tensor_0, Tensor<float, 2> tensor_1, Tensor<float, 2> tensor_2, Tensor<float, 2> tensor_3)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0,0));
    result += tensor_1.load(uint2(0,0));
    result += tensor_2.load(uint2(0,0));
    result += tensor_3.load(uint2(0,0));
    return result;
}

float sum(Tensor<float, 2> tensor_0, Tensor<float, 2> tensor_1, Tensor<float, 2> tensor_2, Tensor<float, 2> tensor_3, Tensor<float, 2> tensor_4)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0,0));
    result += tensor_1.load(uint2(0,0));
    result += tensor_2.load(uint2(0,0));
    result += tensor_3.load(uint2(0,0));
    result += tensor_4.load(uint2(0,0));
    return result;
}

float sum(Tensor<float, 2> tensor_0, Tensor<float, 2> tensor_1, Tensor<float, 2> tensor_2, Tensor<float, 2> tensor_3, Tensor<float, 2> tensor_4, Tensor<float, 2> tensor_5)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0,0));
    result += tensor_1.load(uint2(0,0));
    result += tensor_2.load(uint2(0,0));
    result += tensor_3.load(uint2(0,0));
    result += tensor_4.load(uint2(0,0));
    result += tensor_5.load(uint2(0,0));
    return result;
}

float sum(Tensor<float, 2> tensor_0, Tensor<float, 2> tensor_1, Tensor<float, 2> tensor_2, Tensor<float, 2> tensor_3, Tensor<float, 2> tensor_4, Tensor<float, 2> tensor_5, Tensor<float, 2> tensor_6)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0,0));
    result += tensor_1.load(uint2(0,0));
    result += tensor_2.load(uint2(0,0));
    result += tensor_3.load(uint2(0,0));
    result += tensor_4.load(uint2(0,0));
    result += tensor_5.load(uint2(0,0));
    result += tensor_6.load(uint2(0,0));
    return result;
}

float sum_diff(DiffTensor<float, 2> tensor_0)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0, 0));
    return result;
}

float sum_diff(DiffTensor<float, 2> tensor_0, DiffTensor<float, 2> tensor_1)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0, 0));
    result += tensor_1.load(uint2(0, 0));
    return result;
}

float sum_diff(DiffTensor<float, 2> tensor_0, DiffTensor<float, 2> tensor_1, DiffTensor<float, 2> tensor_2)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0, 0));
    result += tensor_1.load(uint2(0, 0));
    result += tensor_2.load(uint2(0, 0));
    return result;
}

float sum_diff(DiffTensor<float, 2> tensor_0, DiffTensor<float, 2> tensor_1, DiffTensor<float, 2> tensor_2, DiffTensor<float, 2> tensor_3)
{
    float result = 0.0;
    result += tensor_0.load(uint2(0, 0));
    result += tensor_1.load(uint2(0, 0));
    result += tensor_2.load(uint2(0, 0));
    result += tensor_3.load(uint2(0, 0));
    return result;
}

float sum_diff( DiffTensor<float, 2> tensor_0, DiffTensor<float, 2> tensor_1, DiffTensor<float, 2> tensor_2, DiffTensor<float, 2> tensor_3, DiffTensor<float, 2> tensor_4 )
{
    float result = 0.0;
    result += tensor_0.load(uint2(0, 0));
    result += tensor_1.load(uint2(0, 0));
    result += tensor_2.load(uint2(0, 0));
    result += tensor_3.load(uint2(0, 0));
    result += tensor_4.load(uint2(0, 0));
    return result;
}

float sum_diff( DiffTensor<float, 2> tensor_0, DiffTensor<float, 2> tensor_1, DiffTensor<float, 2> tensor_2, DiffTensor<float, 2> tensor_3, DiffTensor<float, 2> tensor_4, DiffTensor<float, 2> tensor_5 )
{
    float result = 0.0;
    result += tensor_0.load(uint2(0, 0));
    result += tensor_1.load(uint2(0, 0));
    result += tensor_2.load(uint2(0, 0));
    result += tensor_3.load(uint2(0, 0));
    result += tensor_4.load(uint2(0, 0));
    result += tensor_5.load(uint2(0, 0));
    return result;
}

float sum_diff( DiffTensor<float, 2> tensor_0, DiffTensor<float, 2> tensor_1, DiffTensor<float, 2> tensor_2, DiffTensor<float, 2> tensor_3, DiffTensor<float, 2> tensor_4, DiffTensor<float, 2> tensor_5, DiffTensor<float, 2> tensor_6 )
{
    float result = 0.0;
    result += tensor_0.load(uint2(0, 0));
    result += tensor_1.load(uint2(0, 0));
    result += tensor_2.load(uint2(0, 0));
    result += tensor_3.load(uint2(0, 0));
    result += tensor_4.load(uint2(0, 0));
    result += tensor_5.load(uint2(0, 0));
    result += tensor_6.load(uint2(0, 0));
    return result;
}
