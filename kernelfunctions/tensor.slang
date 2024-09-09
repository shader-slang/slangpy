

struct Tensor<T, let N : int>
{
    StructuredBuffer<T> buffer;
    uint strides[N];

    T get(int index[N])
    {
        uint idx = 0;
        for (int i = 0; i < N; i++)
        {
            idx += index[i] * strides[i];
        }
        return buffer[idx];
    }
}

struct RWTensor<T, let N : int>
{
    RWStructuredBuffer<T> buffer;
    uint strides[N];

    T get(int index[N])
    {
        uint idx = 0;
        for (int i = 0; i < N; i++)
        {
            idx += index[i] * strides[i];
        }
        return buffer[idx];
    }

    void set(int index[N], T value)
    {
        uint idx = 0;
        for (int i = 0; i < N; i++)
        {
            idx += index[i] * strides[i];
        }
        buffer[idx] = value;
    }
}
