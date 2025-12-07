// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception



int ft_return1(int a)
{
    return a;
}

int ft_return2(int a)
{
    return a;
}

int ft_return3(int a)
{
    return a;
}

int ft_add(int a, int b)
{
    return a + b;
}

int ft_mul(int a, int b)
{
    return a * b;
}

T ft_generic_add<T: IArithmetic>(T a, T b)
{
    return a + b;
}

vector<T, N> ft_scale_vec_generic<T: __BuiltinArithmeticType,let N: int>(vector<T, N> a, T b)
{
    return a * b;
}

T[N] ft_vec_to_array<T : __BuiltinArithmeticType, let N : int>(vector<T, N> v)
{
    T[N] arr;
    for (int i = 0; i < N; i++)
    {
        arr[i] = v[i];
    }
    return arr;
}

T ft_array_sum<T : __BuiltinArithmeticType, let N : int>(T[N] arr)
{
    T sum = {};
    for (int i = 0; i < N; i++)
    {
        sum = sum + arr[i];
    }
    return sum;
}
