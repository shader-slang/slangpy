
interface IContext
{
    uint3 get_thread_id();
    int get_call_id(int dim);
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

int _idx<let N : int>(int[N] index, int[N] stride, int[N] transform) {
    int idx = 0;
    for (int i = 0; i < N; i++) { idx += index[transform[i]] * stride[i]; }
    return idx;
}

int _idx<let N : int>(IContext index, int[N] stride, int[N] transform) {
    int idx = 0;
    for (int i = 0; i < N; i++) { idx += index.get_call_id(transform[i]) * stride[i]; }
    return idx;
}

struct ImplicitCast<N> 
{

}

struct NoneType
{
    void load_primal<T>(IContext context, out T value) { value = T();}
    void store_primal<T>(IContext context, in T value) {};
}

struct ValueType<T>
{
    T value;
    void load_primal(IContext context, out T value) { value = this.value; }
    void store_primal(IContext context, in T value) {};
}

struct StructuredBufferType<T>
{
    StructuredBuffer<T> value;

    void load_primal(IContext context, out T value) { value = this.value[context.get_call_id(0)]; }
    void store_primal(IContext context, in T value) {};

    void load_primal(IContext context, out StructuredBuffer<T> value) { value = this.value; }
    void store_primal(IContext context, in StructuredBuffer<T> value) {};
}

struct RWStructuredBufferType<T> {
    RWStructuredBuffer<T> value;
    void load_primal(IContext context, out T value) { value = this.value[context.get_call_id(0)]; }
    void store_primal(IContext context, in T value) { this.value[context.get_call_id(0)] = value; };

    void load_primal(IContext context, out RWStructuredBuffer<T> value) { value = this.value; }
    void store_primal(IContext context, in RWStructuredBuffer<T> value) {};
}

struct Texture1DType<T> {
    // x map to 0
    int1 toidx(IContext context) { return int1(context.get_call_id(0)); }

    Texture1D<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) {};

    void load_primal(IContext context, out Texture1D<T> value) { value = this.value; }
    void store_primal(IContext context, in Texture1D<T> value) {};
}

struct RWTexture1DType<T> {
    // x map to 0
    int1 toidx(IContext context) { return int1(context.get_call_id(0)); }

    RWTexture1D<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    void load_primal(IContext context, out RWTexture1D<T> value) { value = this.value; }
    void store_primal(IContext context, in RWTexture1D<T> value) {};
}

struct Texture2DType<T> {
    // xy map to 0,1
    int2 toidx(IContext context) { return int2(context.get_call_id(0), context.get_call_id(1)); }

    Texture2D<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) {};

    void load_primal(IContext context, out Texture2D<T> value) { value = this.value; }
    void store_primal(IContext context, in Texture2D<T> value) {};
}
struct RWTexture2DType<T> {
    // xy map to 0,1
    int2 toidx(IContext context) { return int2(context.get_call_id(0), context.get_call_id(1)); }

    RWTexture2D<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    void load_primal(IContext context, out RWTexture2D<T> value) { value = this.value; }
    void store_primal(IContext context, in RWTexture2D<T> value) {};
}

struct Texture1DArrayType<T> {

    // Map with the first index as the x coordinate and the second index as the array slice
    int2 toidx(IContext context) { return int2(context.get_call_id(1), context.get_call_id(0)); }

    Texture1DArray<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) {};

    void load_primal(IContext context, out Texture1DArray<T> value) { value = this.value; }
    void store_primal(IContext context, in Texture1DArray<T> value) {};
}
struct RWTexture1DArrayType<T> {
    // Map with the first index as the x coordinate and the second index as the array slice
    int2 toidx(IContext context) { return int2(context.get_call_id(1), context.get_call_id(0)); }

    RWTexture1DArray<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    void load_primal(IContext context, out RWTexture1DArray<T> value) { value = this.value; }
    void store_primal(IContext context, in RWTexture1DArray<T> value) {};
}

struct Texture3DType<T> {
    // xyz map to 0,1,2
    int3 toidx(IContext context) { return int3(context.get_call_id(0), context.get_call_id(1), context.get_call_id(2)); }

    Texture3D<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) {};

    void load_primal(IContext context, out Texture3D<T> value) { value = this.value; }
    void store_primal(IContext context, in Texture3D<T> value) {};
}
struct RWTexture3DType<T> {
    // xyz map to 0,1,2
    int3 toidx(IContext context) { return int3(context.get_call_id(0), context.get_call_id(1), context.get_call_id(2)); }

    RWTexture3D<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    void load_primal(IContext context, out RWTexture3D<T> value) { value = this.value; }
    void store_primal(IContext context, in RWTexture3D<T> value) {};
}

struct Texture2DArrayType<T> {
    // slice is 0, xy are 1,2
    int3 toidx(IContext context) { return int3(context.get_call_id(1), context.get_call_id(2), context.get_call_id(0)); }

    Texture2DArray<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) {};

    void load_primal(IContext context, out Texture2DArray<T> value) { value = this.value; }
    void store_primal(IContext context, in Texture2DArray<T> value) {};
}
struct RWTexture2DArrayType<T> {
    // slice is 0, xy are 1,2
    int3 toidx(IContext context) { return int3(context.get_call_id(1), context.get_call_id(2), context.get_call_id(0)); }

    RWTexture2DArray<T> value;
    void load_primal(IContext context, out T value) { value = this.value[toidx(context)]; }
    void store_primal(IContext context, in T value) { this.value[toidx(context)] = value;};

    void load_primal(IContext context, out RWTexture2DArray<T> value) { value = this.value; }
    void store_primal(IContext context, in RWTexture2DArray<T> value) {};
}

struct ValueRef<T>
{
    T value;
    void load_primal(IContext context, out T value) { value = this.value; }
    void store_primal(IContext context, in T value) {};
}

struct RWValueRef<T>
{
    RWStructuredBuffer<T> value;
    void load_primal(IContext context, out T value) { value = this.value[0]; }
    void store_primal(IContext context, in T value) { this.value[0] = value; }
}

struct NDBuffer<T, let N : int>
{
    StructuredBuffer<T> buffer;
    int[N] strides;
    int[N] transform;

    T get(int[N] index) { return buffer[_idx(index, strides)]; }
    __subscript(int[N] index)->T { get { return get(index); } }

    void load_primal(IContext context, out T value) { value = buffer[_idx(context, strides, transform)]; }
    void store_primal(IContext context, in T value) {};

    void load_primal(IContext context, out StructuredBuffer<T> value) {
        value = buffer;
    }
    void store_primal(IContext context, in StructuredBuffer<T> value) {};

    void load_primal<let BN : int>(IContext context, out NDBuffer<T, BN> value) { 
        value.buffer = buffer;
        //TODO: Do this properly - it's wrong
        for(int i = 0; i < N; i++) { 
            value.strides[i] = strides[i]; 
            value.transform[i] = transform[i];
        }
    }
    void store_primal<let BN : int>(IContext context, in NDBuffer<T, BN> value) {};

    void load_primal<let VD : int>(IContext context, out vector<T, VD> value) {

        int call_id[N];
        for (int i = 0; i < N - 1; i++) { call_id[i] = context.get_call_id(i); }

        for (int vi = 0; vi < VD; vi++) {
            call_id[N - 1] = vi;
            value[vi] = buffer[_idx(call_id, strides, transform)];
        }
    }

    void store_primal<let VD : int>(IContext context, out vector<T, VD> value) {}
}

struct RWNDBuffer<T, let N : int>
{
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    int[N] transform;

    T get(int[N] index) { return buffer[_idx(index, strides)]; }
    void set(int[N] index, T value) { buffer[_idx(index, strides)] = value; }

    __subscript(int[N] index)->T { get { return get(index); } set { set(index, newValue); } }

    void load_primal(IContext context, out T value) { value = buffer[_idx(context, strides, transform)]; }
    void store_primal(IContext context, in T value) { buffer[_idx(context, strides, transform)] = value; }

    void load_primal<let VD : int>(IContext context, out vector<T, VD> value) {

        int call_id[N];
        for (int i = 0; i < N-1; i++) { call_id[i] = context.get_call_id(i); }
        
        for (int vi = 0; vi < VD; vi++) {
            call_id[N - 1] = vi;
            value[vi] = buffer[_idx(call_id, strides, transform)];
        }
    }

    void store_primal<let VD : int>(IContext context, in vector<T, VD> value) {
        int call_id[N];
        for (int i = 0; i < N - 1; i++) { call_id[i] = context.get_call_id(i); }

        for (int vi = 0; vi < VD; vi++) {
            call_id[N - 1] = vi;
            buffer[_idx(call_id, strides, transform)] = value[vi];
        }
    }
}
