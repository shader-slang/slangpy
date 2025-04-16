// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct ContextND<let N: int>
{
    // Workaround for slang bug #6039
    public uint3 thread_id;
    public int[N] call_id;

    // Conversion to vector gives transpose of the array based coordinate.
    public property vector<int, N> call_id_vec {
        get {
            vector<int, N> result;
            int end = N - 1;
            for (int i = 0; i < N; ++i)
                result[i] = call_id[end-i];
            return result;
        }
    }

    public ContextND<0> map(int)
    {
        return { thread_id };
    }

    public ContextND<M> map<let M : int>(int[M] mapping)
    {
        ContextND<M> result;
        result.thread_id = thread_id;
        for (int i = 0; i < M; ++i)
            result.call_id[i] = call_id[mapping[i]];
        return result;
    }
}
public typealias Context0D = ContextND<0>;
public typealias Context1D = ContextND<1>;
public typealias Context2D = ContextND<2>;
public typealias Context3D = ContextND<3>;
public typealias Context4D = ContextND<4>;
public typealias Context5D = ContextND<5>;

public struct Unknown {}

int _idx<let N : int>(int[N] index, int[N] stride) {
    int idx = 0;
    for (int i = 0; i < N; i++) { idx += index[i] * stride[i]; }
    return idx;
}

int _idx<let N : int>(ContextND<N> index, int[N] stride) {
    int idx = 0;
    for (int i = 0; i < N; i++) { idx += index.call_id[i] * stride[i]; }
    return idx;
}

// _idx_vec handles the fact that the vector form of the coordinate is the transpose
// of the array form.
int _idx_vec<let N : int>(vector<int,N> index, int[N] stride) {
    int idx = 0;
    int end = N - 1;
    for (int i = 0; i < N; i++) { idx += index[end-i] * stride[i]; }
    return idx;
}

public struct ImplicitCast<N>
{

}

public struct NoneType
{
    public void load<T, let N : int>(ContextND<N> context, out T value) { value = T();}
    public void store<T, let N : int>(ContextND<N> context, in T value) {};
}

public struct ValueType<T>
{
    T value;
    public void load(Context0D context, out T value) { value = this.value; }
    public void store(Context0D context, in T value) {};
}

public struct VectorValueType<T,let N: int>
{
    vector<T, N> value;

    public void load(Context0D context, out vector<T,N> value) { value = this.value; }
    public void store(Context0D context, in vector<T,N> value) {};

    public void load(Context1D context, out T value) { value = this.value[context.call_id[0]]; }
    public void store(Context1D context, in T value) {};
}

public struct RangeType
{
    int start;
    int stop;
    int step;
    public void load(Context1D context, out int value) { value = this.start + context.call_id_vec.x * this.step; }
    public void store(Context1D context, in int value) {};
}

public struct StructuredBufferType<T>
{
    StructuredBuffer<T> value;

    public void load(Context0D context, out StructuredBuffer<T> value) { value = this.value; }
    public void store(Context0D context, in StructuredBuffer<T> value) {};
}

public struct RWStructuredBufferType<T> {
    RWStructuredBuffer<T> value;

    public void load(Context0D context, out RWStructuredBuffer<T> value) { value = this.value; }
    public void store(Context0D context, in RWStructuredBuffer<T> value) {};
}

public struct ByteAddressBufferType
{
    ByteAddressBuffer value;

    public void load(Context0D context, out ByteAddressBuffer value) { value = this.value; }
    public void store(Context0D context, in ByteAddressBuffer value) {};
}

public struct RWByteAddressBufferType {
    RWByteAddressBuffer value;

    public void load(Context0D context, out RWByteAddressBuffer value) { value = this.value; }
    public void store(Context0D context, in RWByteAddressBuffer value) {};
}

public struct Texture1DType<T : ITexelElement> {
    Texture1D<T> value;
    public void load(Context1D context, out T value) { value = this.value[context.call_id_vec]; }
    public void store(Context1D context, in T value) {};

    public void load(Context0D context, out Texture1D<T> value) { value = this.value; }
    public void store(Context0D context, in Texture1D<T> value) {};
}

public struct RWTexture1DType<T : ITexelElement> {
    RWTexture1D<T> value;
    public void load(Context1D context, out T value) { value = this.value[context.call_id_vec]; }
    public void store(Context1D context, in T value) { this.value[context.call_id_vec] = value;};

    public void load(Context0D context, out RWTexture1D<T> value) { value = this.value; }
    public void store(Context0D context, in RWTexture1D<T> value) {};
}

public struct Texture2DType<T : ITexelElement> {
    Texture2D<T> value;
    public void load(Context2D context, out T value) { value = this.value[context.call_id_vec]; }
    public void store(Context2D context, in T value) {};

    public void load(Context0D context, out Texture2D<T> value) { value = this.value; }
    public void store(Context0D context, in Texture2D<T> value) {};
}
public struct RWTexture2DType<T : ITexelElement> {
    RWTexture2D<T> value;
    public void load(Context2D context, out T value) { value = this.value[context.call_id_vec]; }
    public void store(Context2D context, in T value) { this.value[context.call_id_vec] = value;};

    public void load(Context0D context, out RWTexture2D<T> value) { value = this.value; }
    public void store(Context0D context, in RWTexture2D<T> value) {};
}

public struct Texture1DArrayType<T : ITexelElement> {

    // Map with the first index as the x coordinate and the second index as the array slice
    int2 toidx(Context2D context) { return context.call_id_vec; }

    Texture1DArray<T> value;
    public void load(Context2D context, out T value) { value = this.value[toidx(context)]; }
    public void store(Context2D context, in T value) {};

    public void load(Context0D context, out Texture1DArray<T> value) { value = this.value; }
    public void store(Context0D context, in Texture1DArray<T> value) {};
}
public struct RWTexture1DArrayType<T : ITexelElement> {
    // Map with the first index as the x coordinate and the second index as the array slice
    int2 toidx(Context2D context) { return context.call_id_vec; }

    RWTexture1DArray<T> value;
    public void load(Context2D context, out T value) { value = this.value[toidx(context)]; }
    public void store(Context2D context, in T value) { this.value[toidx(context)] = value;};

    public void load(Context0D context, out RWTexture1DArray<T> value) { value = this.value; }
    public void store(Context0D context, in RWTexture1DArray<T> value) {};
}

public struct Texture3DType<T : ITexelElement> {
    Texture3D<T> value;
    public void load(Context3D context, out T value) { value = this.value[context.call_id_vec]; }
    public void store(Context3D context, in T value) {};

    public void load(Context0D context, out Texture3D<T> value) { value = this.value; }
    public void store(Context0D context, in Texture3D<T> value) {};
}
public struct RWTexture3DType<T : ITexelElement> {
    RWTexture3D<T> value;
    public void load(Context3D context, out T value) { value = this.value[context.call_id_vec]; }
    public void store(Context3D context, in T value) { this.value[context.call_id_vec] = value;};

    public void load(Context0D context, out RWTexture3D<T> value) { value = this.value; }
    public void store(Context0D context, in RWTexture3D<T> value) {};
}

public struct Texture2DArrayType<T : ITexelElement> {
    // slice is 0, y is 1, x is 2
    int3 toidx(Context3D context) { return context.call_id_vec; }

    Texture2DArray<T> value;
    public void load(Context3D context, out T value) { value = this.value[toidx(context)]; }
    public void store(Context3D context, in T value) {};

    public void load(Context0D context, out Texture2DArray<T> value) { value = this.value; }
    public void store(Context0D context, in Texture2DArray<T> value) {};
}
public struct RWTexture2DArrayType<T : ITexelElement> {
    // slice is 0, xy are 1,2
    int3 toidx(Context3D context) { return context.call_id_vec; }

    RWTexture2DArray<T> value;
    public void load(Context3D context, out T value) { value = this.value[toidx(context)]; }
    public void store(Context3D context, in T value) { this.value[toidx(context)] = value;};

    public void load(Context0D context, out RWTexture2DArray<T> value) { value = this.value; }
    public void store(Context0D context, in RWTexture2DArray<T> value) {};
}

public struct RaytracingAccelerationStructureType
{
    RaytracingAccelerationStructure value;
    public void load(Context0D context, out RaytracingAccelerationStructure value) { value = this.value; }
    public void store(Context0D context, in RaytracingAccelerationStructure value) {};
}

public struct SamplerStateType
{
    public SamplerState value;
    public void load(Context0D context, out SamplerState value) { value = this.value; }
    public void store(Context0D context, in SamplerState value) {};
}

public struct ValueRef<T>
{
    T value;
    public void load(Context0D context, out T value) { value = this.value; }
    public void store(Context0D context, in T value) {};
}

public struct RWValueRef<T>
{
    RWStructuredBuffer<T> value;
    public void load(Context0D context, out T value) { value = this.value[0]; }
    public void store(Context0D context, in T value) { this.value[0] = value; }
}

public struct NDBuffer<T, let N : int>
{
    public StructuredBuffer<T> buffer;
    public int offset;
    public int[N] strides;
    public int[N] shape;

    public T get(int[N] index) { return buffer[offset + _idx(index, strides)]; }
    //public T get(vector<int,N> index) { return buffer[offset + _idx(index, strides)]; }

    //public __subscript(int[N] index)->T { get { return get(index); } }
    public __subscript(vector<int,N> index)->T { get { return buffer[offset + _idx_vec(index, strides)]; } }

    public void load(ContextND<N> context, out T value) { value = get(context.call_id); }
    public void store(ContextND<N> context, in T value) {};

    public void load(Context0D context, out NDBuffer<T,N> value) {
        value = this;
    }
    public void store(Context0D context, in NDBuffer<T, N> value) {};

    public void load(Context0D context, out StructuredBuffer<T> value) {
        value = buffer;
    }
    public void store(Context0D context, in StructuredBuffer<T> value) {};

    public void load<let VD : int>(ContextND<N - 1> context, out vector<T, VD> value) {

        int call_id[N];
        for (int i = 0; i < N - 1; i++) { call_id[i] = context.call_id[i]; }

        for (int vi = 0; vi < VD; vi++) {
            call_id[N - 1] = vi;
            value[vi] = buffer[offset + _idx(call_id, strides)];
        }
    }

    public void store<let VD : int>(ContextND<N - 1> context, out vector<T, VD> value) {}
}

public struct RWNDBuffer<T, let N : int>
{
    public RWStructuredBuffer<T> buffer;
    public int offset;
    public int[N] strides;
    public int[N] shape;

    public T get(int[N] index) { return buffer[offset + _idx(index, strides)]; }
    //public T get(vector<int,N> index) { return buffer[offset + _idx(index, strides)]; }

    public void set(int[N] index, T value) { buffer[offset + _idx(index, strides)] = value; }
    //public void set(vector<int,N> index, T value) { buffer[offset + _idx(index, strides)] = value; }

    //public __subscript(int[N] index)->T {
    //    get { return get(index); }
    //    [nonmutating]
    //    set { set(index, newValue); }
    //}

    public __subscript(vector<int,N> index)->T {
        get { return buffer[offset + _idx_vec(index, strides)]; }
        [nonmutating]
        set { buffer[offset + _idx_vec(index, strides)] = newValue; }
    }

    public void load(ContextND<N> context, out T value) { value = get(context.call_id); }
    public void store(ContextND<N> context, in T value) { set(context.call_id, value); }

    public void load(Context0D context, out RWStructuredBuffer<T> value) {
        value = buffer;
    }

    public void load(Context0D context, out RWNDBuffer<T, N> value) {
        value = this;
    }
    public void store(Context0D context, in RWNDBuffer<T, N> value) {};

    public void load<let VD : int>(ContextND<N - 1> context, out vector<T, VD> value) {
        int call_id[N];
        for (int i = 0; i < N-1; i++) { call_id[i] = context.call_id[i]; }

        for (int vi = 0; vi < VD; vi++) {
            call_id[N - 1] = vi;
            value[vi] = get(call_id);
        }
    }

    public void store<let VD : int>(ContextND<N - 1> context, in vector<T, VD> value) {
        int call_id[N];
        for (int i = 0; i < N - 1; i++) { call_id[i] = context.call_id[i]; }

        for (int vi = 0; vi < VD; vi++) {
            call_id[N - 1] = vi;
            set(call_id, value[vi]);
        }
    }
}

namespace impl
{

    // Utility interface to check if a python type is safe to replace by the target
    // slang type. This works around mismatches between the type systems: Python e.g.
    // only knows float or int, but slang has a much wider array of scalar types;
    // integers alone have 8 different types.
    // This gets even worse when lists are thrown into the mix, and trying to pass
    // e.g. [1, 2, 3] to uint16_t[3] would fail if the ValueType picks the default
    // of int[3] to pass to slang. We could attempt to check this in python, but
    // this would require deep slang type inspection and would inevitably be brittle
    //
    // To solve this, we check if the python type (e.g. float[3]) type can be converted
    // safely to the slang type using slang generic constraints.
    // If type B implements IConvertibleFrom<A>, then we can safely convert from A to B
    interface IConvertibleFrom<From> {}

    // A python int can be passed to any slang integer scalar type
    extension<T : __BuiltinIntegerType> T : IConvertibleFrom<int> {}

    // A python float can be passed to any slang float scalar type
    extension<T : __BuiltinFloatingPointType> T : IConvertibleFrom<float> {}

    // A python array can be passed to any slang array of the same length, as
    // long as their element types can be safely passed
    extension<S, T : IConvertibleFrom<S>, let N : int, ArrT : ISizedArray<T, N>> ArrT : IConvertibleFrom<Array<S, N>> {}

    // A vector is convertible to another vector of the same length who's element types are convertible
    extension<S, T : IConvertibleFrom<S>, let N : int> vector<T,N> : IConvertibleFrom<vector<S, N>> {}

    // This function allows us to check if type A is convertible to type B.
    // We attempt to specialize allowedConversionWitness with types [A, B]
    // If it succeeds, the interface requirements are satisfied.
    struct AllowedConversionWitness<From, To : IConvertibleFrom<From>> {}

}
