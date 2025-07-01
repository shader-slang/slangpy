// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#ifndef INPUT_TENSOR_COUNT
#define INPUT_TENSOR_COUNT 1
#endif

#define INDEX_MODE_ARRAY 0
#define INDEX_MODE_VECTOR 1
#define INDEX_MODE_SUBSCRIPT 2

#ifndef INDEX_MODE
#define INDEX_MODE INDEX_MODE_ARRAY
#endif

#define READ_PATTERN_SINGLE 0
#define READ_PATTERN_REGION 1

#ifndef READ_PATTERN
#define READ_PATTERN READ_PATTERN_SINGLE
#endif

#define READ_MODE_DIRECT 0
#define READ_MODE_INDIRECT 1

#ifndef READ_MODE
#define READ_MODE READ_MODE_DIRECT
#endif

interface ITensor<T, let D : int>
{
    property int[D] shape;
    T get(int idx[D]);
    T getv(vector<int, D> idx);
}

interface IRWTensor<T, let D : int> : ITensor<T, D>
{
    void set(int idx[D], T value);
    void setv(vector<int, D> idx, T value);
}

struct StridedLayout<let D : int>
{
    int offset;
    int[D] strides;

    int at(int[D] idx)
    {
        int result = offset;
        for (int i = 0; i < D; ++i)
            result += strides[i] * idx[i];
        return result;
    }

    int at_vec(vector<int, D> idx)
    {
        int result = offset;
        for (int i = 0; i < D; ++i)
            result += strides[i] * idx[i];
        return result;
    }
}

namespace detail {
int at<let D : int>(int[D] idx, int offset, int[D] strides)
{
    int result = offset;
    for (int i = 0; i < D; ++i)
        result += strides[i] * idx[i];
    return result;
}
int at_vec<let D : int>(vector<int, D> idx, int offset, int[D] strides)
{
    int result = offset;
    for (int i = 0; i < D; ++i)
        result += strides[i] * idx[i];
    return result;
}
int[D] makeIndex<let D : int, each T : IInteger>(expand each T indices)
{
    // We can't currently specify a type constraint that we expect exactly D Ts, so
    // static_assert is needed
    static_assert(countof(T) == D, "Number of indices does not match Tensor dimensionality");
    int[D] idxVec;
    int i = 0;
    expand idxVec[i++] = (each indices).toInt();
    return idxVec;
}
}

struct Tensor<T, let D : int> : ITensor<T, D> {
    StructuredBuffer<T> buffer;
    StridedLayout<D> layout;
    int[D] shape;

    T get(int[D] idx) { return buffer[layout.at(idx)]; }
    T getv(vector<int, D> idx) { return buffer[layout.at_vec(idx)]; }
}

struct RWTensor<T, let D : int> : IRWTensor<T, D> {
    RWStructuredBuffer<T> buffer;
    StridedLayout<D> layout;
    int[D] shape;

    T get(int[D] idx) { return buffer[layout.at(idx)]; }
    T getv(vector<int, D> idx) { return buffer[layout.at_vec(idx)]; }
    void set(int[D] idx, T value) { buffer[layout.at(idx)] = value; }
    void setv(vector<int, D> idx, T value) { buffer[layout.at_vec(idx)] = value; }
}

extension<T, let D : int, TensorType : ITensor<T, D>> TensorType
{
    __generic<each Is : IInteger> __subscript(expand each Is indices)->T
    {
        get {
            return get(detail::makeIndex<D>(expand each indices));
        }
    }
}

struct Data {
    Tensor<float, 2> input[INPUT_TENSOR_COUNT];
    RWTensor<float, 2> output;
    uint input_tensor_count;
    StructuredBuffer<uint> index_buffer;
    uint index_count;

    float fetch(int index, int2 texel)
    {
#if INDEX_MODE == INDEX_MODE_ARRAY
        return input[index].get({ texel.x, texel.y });
#elif INDEX_MODE == INDEX_MODE_VECTOR
        return input[index].getv(int2(texel.x, texel.y));
#elif INDEX_MODE == INDEX_MODE_SUBSCRIPT
        return input[index][texel.x, texel.y];
#endif
    }
};

#if INDEX_MODE == INDEX_MODE_ARRAY
#define READ_DIRECT(i, x, y) data.input[i].get({ x, y })
#elif INDEX_MODE == INDEX_MODE_VECTOR
#define READ_DIRECT(i, x, y) data.input[i].getv(int2(x, y))
#elif INDEX_MODE == INDEX_MODE_SUBSCRIPT
#define READ_DIRECT(i, x, y) data.input[i][x, y]
#endif

#if READ_MODE == READ_MODE_DIRECT
#define READ(i, x, y) READ_DIRECT(i, x, y)
#elif READ_MODE == READ_MODE_INDIRECT
#define READ(i, x, y) data.fetch(i, int2(x, y))
#endif

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, ParameterBlock<Data> data)
{
    float result = 0.0;
    // for (int i = 0; i < data.input_tensor_count; ++i) {
    for (int i = 0; i < data.index_count; ++i) {
        uint ti = data.index_buffer[i];
#if READ_PATTERN == READ_PATTERN_SINGLE
        result += READ(ti, tid.x, tid.y);
#elif READ_PATTERN == READ_PATTERN_REGION
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                result += READ(ti, (tid.x + j) % 1024, (tid.y + k) % 1024);
            }
        }
#endif
    }
    data.output.set( { tid.x, tid.y }, result);
}
