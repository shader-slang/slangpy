
#ifndef INPUT_TENSOR_COUNT
#define INPUT_TENSOR_COUNT 1
#endif

#define INDEX_MODE_ARRAY 0
#define INDEX_MODE_VECTOR 1
#define INDEX_MODE_SUBSCRIPT 2

#ifndef INDEX_MODE
#define INDEX_MODE INDEX_MODE_ARRAY
#endif

#define ACCESS_MODE_SINGLE 0
#define ACCESS_MODE_REGION 1

#ifndef ACCESS_MODE
#define ACCESS_MODE ACCESS_MODE_SINGLE
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

    // int offset;
    // int[D] strides;

    // T get(int[D] idx) { return buffer[detail::at(idx, offset, strides)]; }
    // T getv(vector<int, D> idx) { return buffer[detail::at_vec(idx, offset, strides)]; }
}

struct RWTensor<T, let D : int> : IRWTensor<T, D> {
    RWStructuredBuffer<T> buffer;
    StridedLayout<D> layout;
    int[D] shape;

    T get(int[D] idx) { return buffer[layout.at(idx)]; }
    T getv(vector<int, D> idx) { return buffer[layout.at_vec(idx)]; }
    void set(int[D] idx, T value) { buffer[layout.at(idx)] = value; }
    void setv(vector<int, D> idx, T value) { buffer[layout.at_vec(idx)] = value; }

    // int offset;
    // int[D] strides;

    // T get(int[D] idx) { return buffer[detail::at(idx, offset, strides)]; }
    // T getv(vector<int, D> idx) { return buffer[detail::at_vec(idx, offset, strides)]; }
    // void set(int[D] idx, T value) { buffer[detail::at(idx, offset, strides)] = value; }
    // void setv(vector<int, D> idx, T value) { buffer[detail::at_vec(idx, offset, strides)] = value; }
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
    Tensor<float, 3> input[INPUT_TENSOR_COUNT];
    RWTensor<float, 3> output;

    float3 fetch(int index, int2 texel)
    {
        texel.x = clamp(texel.x, 0, input[index].shape[1] - 1);
        texel.y = clamp(texel.y, 0, input[index].shape[0] - 1);

        float3 val;
        val.x = input[index][texel.y, texel.x, 0];
        val.y = input[index][texel.y, texel.x, 1];
        val.z = input[index][texel.y, texel.x, 2];
        return val;
    }
};

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, ParameterBlock<Data> data)
{
#if INDEX_MODE == INDEX_MODE_ARRAY
    float3 result = float3(0.0);
    for (int i = 0; i < INPUT_TENSOR_COUNT; ++i) {
#if ACCESS_MODE == ACCESS_MODE_SINGLE
        result.x += data.input[i].get( { tid.x, tid.y, 0 });
        result.y += data.input[i].get( { tid.x, tid.y, 1 });
        result.z += data.input[i].get( { tid.x, tid.y, 2 });
#elif ACCESS_MODE == ACCESS_MODE_REGION
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                result += data.fetch(i, int2(tid.x + j, tid.y + k));
                // result.x += data.input[i].get( { (tid.x + j) % 1024, (tid.y + k) % 1024, 0 });
                // result.y += data.input[i].get( { (tid.x + j) % 1024, (tid.y + k) % 1024, 1 });
                // result.z += data.input[i].get( { (tid.x + j) % 1024, (tid.y + k) % 1024, 2 });
            }
        }
#endif
    }
    data.output.set( { tid.x, tid.y, 0 }, result.x);
    data.output.set( { tid.x, tid.y, 1 }, result.y);
    data.output.set( { tid.x, tid.y, 2 }, result.z);
#elif INDEX_MODE == INDEX_MODE_VECTOR
    float3 result = float3(0.0);
    for (int i = 0; i < INPUT_TENSOR_COUNT; ++i) {
#if ACCESS_MODE == ACCESS_MODE_SINGLE
        result.x += data.input[i].getv(int3(tid.x, tid.y, 0));
        result.y += data.input[i].getv(int3(tid.x, tid.y, 1));
        result.z += data.input[i].getv(int3(tid.x, tid.y, 2));
#elif ACCESS_MODE == ACCESS_MODE_REGION
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                result.x += data.input[i].getv(int3((tid.x + j) % 1024, (tid.y + k) % 1024, 0));
                result.y += data.input[i].getv(int3((tid.x + j) % 1024, (tid.y + k) % 1024, 1));
                result.z += data.input[i].getv(int3((tid.x + j) % 1024, (tid.y + k) % 1024, 2));
            }
        }
#endif
    }
    data.output.setv(int3(tid.x, tid.y, 0), result.x);
    data.output.setv(int3(tid.x, tid.y, 1), result.y);
    data.output.setv(int3(tid.x, tid.y, 2), result.z);
#elif INDEX_MODE == INDEX_MODE_SUBSCRIPT
    float3 result = float3(0.0);
    for (int i = 0; i < INPUT_TENSOR_COUNT; ++i) {
#if ACCESS_MODE == ACCESS_MODE_SINGLE
        result.x += data.input[i][tid.x, tid.y, 0];
        result.y += data.input[i][tid.x, tid.y, 1];
        result.z += data.input[i][tid.x, tid.y, 2];
#elif ACCESS_MODE == ACCESS_MODE_REGION
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                result.x += data.input[i][(tid.x + j) % 1024, (tid.y + k) % 1024, 0];
                result.y += data.input[i][(tid.x + j) % 1024, (tid.y + k) % 1024, 1];
                result.z += data.input[i][(tid.x + j) % 1024, (tid.y + k) % 1024, 2];
            }
        }
#endif
    }
    data.output.setv(int3(tid.x, tid.y, 0), result.x);
    data.output.setv(int3(tid.x, tid.y, 1), result.y);
    data.output.setv(int3(tid.x, tid.y, 2), result.z);
#endif
}
