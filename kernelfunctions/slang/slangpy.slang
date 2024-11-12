module slangpy;

public interface IContext
{
    public uint3 get_thread_id();
    public int get_call_id(int dim);
}

public struct ContextND<let N: int> : IContext
{
    uint3 thread_id;
    int[N] call_id;

    public uint3 get_thread_id() { return thread_id; }
    public int get_call_id(int dim) { return call_id[dim]; }
}
public struct Context0D : IContext
{
    uint3 thread_id;
    public uint3 get_thread_id() { return thread_id; }
    public int get_call_id(int dim) { return 0; }
}

public struct Unknown {}

public ContextND<N> ctx<let N: int>(IContext other, int[N] mapping) {
    ContextND<N> context;
    context.thread_id = other.get_thread_id();
    [unroll]
    for (int i = 0; i < N; i++) { context.call_id[i] = other.get_call_id(mapping[i]); }
    return context;
}
public Context0D ctx(IContext other, int mapping) {
    Context0D context;
    context.thread_id = other.get_thread_id();
    return context;
}

int _idx<let N : int>(int[N] index, int[N] stride) {
    int idx = 0;
    for (int i = 0; i < N; i++) { idx += index[i] * stride[i]; }
    return idx;
}

int _idx<let N : int>(IContext index, int[N] stride) {
    int idx = 0;
    for (int i = 0; i < N; i++) { idx += index.get_call_id(i) * stride[i]; }
    return idx;
}

public struct ImplicitCast<N> 
{

}

public struct NoneType
{
    public void load_primal<T>(IContext context, out T value) { value = T();}
    public void store_primal<T>(IContext context, in T value) {};
}

public struct ValueType<T>
{
    T value;
    public void load_primal(IContext context, out T value) { value = this.value; }
    public void store_primal(IContext context, in T value) {};
}

public struct StructuredBufferType<T>
{
    StructuredBuffer<T> value;

    public void load_primal(IContext context, out T value) { value = this.value[context.get_call_id(0)]; }
    public void store_primal(IContext context, in T value) {};

    public void load_primal(IContext context, out StructuredBuffer<T> value) { value = this.value; }
    public void store_primal(IContext context, in StructuredBuffer<T> value) {};
}

public struct RWStructuredBufferType<T> {
    RWStructuredBuffer<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[context.get_call_id(0)]; }
    public void store_primal(IContext context, in T value) { this.value[context.get_call_id(0)] = value; };

    public void load_primal(IContext context, out RWStructuredBuffer<T> value) { value = this.value; }
    public void store_primal(IContext context, in RWStructuredBuffer<T> value) {};
}

public struct Texture1DType<T> {
    // x map to 0
    int1 toidx(IContext context) { return int1(context.get_call_id(0)); }

    Texture1D<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) {};

    public void load_primal(IContext context, out Texture1D<T> value) { value = this.value; }
    public void store_primal(IContext context, in Texture1D<T> value) {};
}

public struct RWTexture1DType<T> {
    // x map to 0
    int1 toidx(IContext context) { return int1(context.get_call_id(0)); }

    RWTexture1D<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    public void load_primal(IContext context, out RWTexture1D<T> value) { value = this.value; }
    public void store_primal(IContext context, in RWTexture1D<T> value) {};
}

public struct Texture2DType<T> {
    // xy map to 0,1
    int2 toidx(IContext context) { return int2(context.get_call_id(0), context.get_call_id(1)); }

    Texture2D<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) {};

    public void load_primal(IContext context, out Texture2D<T> value) { value = this.value; }
    public void store_primal(IContext context, in Texture2D<T> value) {};
}
public struct RWTexture2DType<T> {
    // xy map to 0,1
    int2 toidx(IContext context) { return int2(context.get_call_id(0), context.get_call_id(1)); }

    RWTexture2D<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    public void load_primal(IContext context, out RWTexture2D<T> value) { value = this.value; }
    public void store_primal(IContext context, in RWTexture2D<T> value) {};
}

public struct Texture1DArrayType<T> {

    // Map with the first index as the x coordinate and the second index as the array slice
    int2 toidx(IContext context) { return int2(context.get_call_id(1), context.get_call_id(0)); }

    Texture1DArray<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) {};

    public void load_primal(IContext context, out Texture1DArray<T> value) { value = this.value; }
    public void store_primal(IContext context, in Texture1DArray<T> value) {};
}
public struct RWTexture1DArrayType<T> {
    // Map with the first index as the x coordinate and the second index as the array slice
    int2 toidx(IContext context) { return int2(context.get_call_id(1), context.get_call_id(0)); }

    RWTexture1DArray<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    public void load_primal(IContext context, out RWTexture1DArray<T> value) { value = this.value; }
    public void store_primal(IContext context, in RWTexture1DArray<T> value) {};
}

public struct Texture3DType<T> {
    // xyz map to 0,1,2
    int3 toidx(IContext context) { return int3(context.get_call_id(0), context.get_call_id(1), context.get_call_id(2)); }

    Texture3D<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) {};

    public void load_primal(IContext context, out Texture3D<T> value) { value = this.value; }
    public void store_primal(IContext context, in Texture3D<T> value) {};
}
public struct RWTexture3DType<T> {
    // xyz map to 0,1,2
    int3 toidx(IContext context) { return int3(context.get_call_id(0), context.get_call_id(1), context.get_call_id(2)); }

    RWTexture3D<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    public void load_primal(IContext context, out RWTexture3D<T> value) { value = this.value; }
    public void store_primal(IContext context, in RWTexture3D<T> value) {};
}

public struct Texture2DArrayType<T> {
    // slice is 0, xy are 1,2
    int3 toidx(IContext context) { return int3(context.get_call_id(1), context.get_call_id(2), context.get_call_id(0)); }

    Texture2DArray<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) {};

    public void load_primal(IContext context, out Texture2DArray<T> value) { value = this.value; }
    public void store_primal(IContext context, in Texture2DArray<T> value) {};
}
public struct RWTexture2DArrayType<T> {
    // slice is 0, xy are 1,2
    int3 toidx(IContext context) { return int3(context.get_call_id(1), context.get_call_id(2), context.get_call_id(0)); }

    RWTexture2DArray<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    public void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    public void load_primal(IContext context, out RWTexture2DArray<T> value) { value = this.value; }
    public void store_primal(IContext context, in RWTexture2DArray<T> value) {};
}

public struct ValueRef<T>
{
    T value;
    public void load_primal(IContext context, out T value) { value = this.value; }
    public void store_primal(IContext context, in T value) {};
}

public struct RWValueRef<T>
{
    RWStructuredBuffer<T> value;
    public void load_primal(IContext context, out T value) { value = this.value[0]; }
    public void store_primal(IContext context, in T value) { this.value[0] = value; }
}

public struct NDBuffer<T, let N : int>
{
    StructuredBuffer<T> buffer;
    int[N] strides;

    public T get(int[N] index) { return buffer[_idx(index, strides)]; }
    public __subscript(int[N] index)->T { get { return get(index); } }

    public void load_primal(IContext context, out T value) { value = buffer[_idx(context, strides)]; }
    public void store_primal(IContext context, in T value) {};

    public void load_primal(IContext context, out NDBuffer<T,N> value) {
        value = this;
    }
    public void store_primal(IContext context, in NDBuffer<T, N> value) {};

    public void load_primal(IContext context, out StructuredBuffer<T> value) {
        value = buffer;
    }
    public void store_primal(IContext context, in StructuredBuffer<T> value) {};

    public void load_primal<let VD : int>(IContext context, out vector<T, VD> value) {

        int call_id[N];
        for (int i = 0; i < N - 1; i++) { call_id[i] = context.get_call_id(i); }

        for (int vi = 0; vi < VD; vi++) {
            call_id[N - 1] = vi;
            value[vi] = buffer[_idx(call_id, strides)];
        }
    }

    public void store_primal<let VD : int>(IContext context, out vector<T, VD> value) {}
}

public struct RWNDBuffer<T, let N : int>
{
    RWStructuredBuffer<T> buffer;
    int[N] strides;

    public T get(int[N] index) { return buffer[_idx(index, strides)]; }
    public void set(int[N] index, T value) { buffer[_idx(index, strides)] = value; }

    public __subscript(int[N] index)->T { get { return get(index); } set { set(index, newValue); } }

    public void load_primal(IContext context, out T value) { value = buffer[_idx(context, strides)]; }
    public void store_primal(IContext context, in T value) { buffer[_idx(context, strides)] = value; }

    public void load_primal(IContext context, out RWNDBuffer<T, N> value) {
        value = this;
    }
    public void store_primal(IContext context, in RWNDBuffer<T, N> value) {};

    public void load_primal<let VD : int>(IContext context, out vector<T, VD> value) {
        int call_id[N];
        for (int i = 0; i < N-1; i++) { call_id[i] = context.get_call_id(i); }
        
        for (int vi = 0; vi < VD; vi++) {
            call_id[N - 1] = vi;
            value[vi] = buffer[_idx(call_id, strides)];
        }
    }

    public void store_primal<let VD : int>(IContext context, in vector<T, VD> value) {
        int call_id[N];
        for (int i = 0; i < N - 1; i++) { call_id[i] = context.get_call_id(i); }

        for (int vi = 0; vi < VD; vi++) {
            call_id[N - 1] = vi;
            buffer[_idx(call_id, strides)] = value[vi];
        }
    }
}
