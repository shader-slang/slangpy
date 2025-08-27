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
        result += tensors[i].getv(tid);
    }
    return result;
}

struct TensorList<let N : int>
{
    RWTensor<float, 2> tensors[N];

    float fetch(int tensor_index, uint2 index)
    {
        return tensors[tensor_index].getv(index);
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
