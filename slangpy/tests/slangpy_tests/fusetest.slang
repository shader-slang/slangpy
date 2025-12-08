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

int ft_sub(int a, int b)
{
    return a - b;
}

int ft_div(int a, int b)
{
    return a / b;
}

int ft_negate(int a)
{
    return -a;
}

float ft_add_float(float a, float b)
{
    return a + b;
}

float ft_mul_float(float a, float b)
{
    return a * b;
}

int ft_max(int a, int b)
{
    return a > b ? a : b;
}

int ft_min(int a, int b)
{
    return a < b ? a : b;
}

T ft_generic_mul<T: IArithmetic>(T a, T b)
{
    return a * b;
}

// Float operations for tensor tests
float ft_sub_float(float a, float b)
{
    return a - b;
}

float ft_div_float(float a, float b)
{
    return a / b;
}

float ft_square_float(float a)
{
    return a * a;
}

float ft_sqrt_float(float a)
{
    return sqrt(a);
}

float ft_negate_float(float a)
{
    return -a;
}

struct FloatContainer: IDifferentiable
{
    float value;
};

float ft_struct_add(FloatContainer a, FloatContainer b)
{
    return a.value + b.value;
}

FloatContainer ft_struct_mul_scalar(FloatContainer a, float scalar)
{
    FloatContainer result;
    result.value = a.value * scalar;
    return result;
}

float ft_struct_get_value(FloatContainer a)
{
    return a.value;
}

struct TypeWithMethod: IDifferentiable
{
    float value;

    float multiply_by_scalar(float scalar)
    {
        return value * scalar;
    }

    static float static_multiply_by_scalar(TypeWithMethod instance, float scalar)
    {
        return instance.value * scalar;
    }
};
