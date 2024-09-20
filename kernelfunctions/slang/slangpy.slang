
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
}

struct RWStructuredBufferType<T> {
    RWStructuredBuffer<T> value;
    void load_primal(IContext context, out T value) { value = this.value[context.get_call_id(0)]; }
    void store_primal(IContext context, in T value) { this.value[context.get_call_id(0)] = value; };
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
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    int[N] transform;

    T get(int[N] index) { return buffer[_idx(index, strides)]; }
    __subscript(int[N] index)->T { get { return get(index); } }

    void load_primal(IContext context, out T value) { value = buffer[_idx(context, strides, transform)]; }
    void store_primal(IContext context, in T value) {};

    void load_primal<let VD : int>(IContext context, out vector<T, VD> value) {
        int idx = 0;
        for (int i = 0; i < N-1; i++) { idx += context.get_call_id(transform[i]) * strides[i]; }

        for (int vi = 0; vi < VD; vi++) {
            value[vi] = buffer[idx];
            idx += strides[N-1];
        }
    }

    void store_primal<let VD : int>(IContext context, out vector<T, VD> value) {}
}

struct RWTensorBuffer<T, let N : int>
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
        int idx = 0;
        for (int i = 0; i < N - 1; i++) { idx += context.get_call_id(transform[i]) * strides[i]; }

        for (int vi = 0; vi < VD; vi++) {
            value[vi] = buffer[idx];
            idx += strides[N - 1];
        }
    }

    void store_primal<let VD : int>(IContext context, in vector<T, VD> value) {
        int idx = 0;
        for (int i = 0; i < N - 1; i++) { idx += context.get_call_id(transform[i]) * strides[i]; }

        for (int vi = 0; vi < VD; vi++) {
            buffer[idx] = value[vi];
            idx += strides[N - 1];
        }
    }
}
