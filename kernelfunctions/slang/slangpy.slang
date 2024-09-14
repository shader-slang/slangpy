
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

interface ICallData<PT, DT>
{
    associatedtype primal_type;
    associatedtype derivative_type;

    void load_primal(IContext context, out PT value);
    void store_primal(IContext context, in PT value);
    void load_derivative(IContext context, out DT value);
    void store_derivative(IContext context, in DT value);
}

interface IValueCallData<T>
{
    associatedtype primal_type;

    void load_primal(IContext context, out T value);
    void store_primal(IContext context, in T value);
}

struct NoneType<T> : IValueCallData<T>
{
    typealias primal_type = T;
    void load_primal(IContext context, out primal_type value) { value = primal_type();}
    void store_primal(IContext context, in T value) {};
}

struct ValueType<T> : IValueCallData<T>
{
    typealias primal_type = T;
    primal_type value;
    void load_primal(IContext context, out primal_type value) { value = this.value; }
    void store_primal(IContext context, in T value) {};
}

struct ValueRef<T> : IValueCallData<T>
{
    typealias primal_type = T;
    primal_type value;
    void load_primal(IContext context, out primal_type value) { value = this.value; }
    void store_primal(IContext context, in T value) {};
}

struct RWValueRef<T> : IValueCallData<T>
{
    typealias primal_type = T;
    RWStructuredBuffer<primal_type> value;
    void load_primal(IContext context, out primal_type value) { value = this.value[0]; }
    void store_primal(IContext context, in primal_type value) { this.value[0] = value; }
}

struct BaseDiffPair<PT, DT, PrimalValue: IValueCallData<PT>, DerivativeValue: IValueCallData<DT>> : ICallData<PT, DT>
{
    typealias primal_type = PrimalValue::primal_type;
    typealias derivative_type = DerivativeValue::primal_type;
    PrimalValue primal;
    DerivativeValue derivative;
    void load_primal(IContext context, out PT value) { primal.load_primal(context, value); }
    void store_primal(IContext context, in PT value) { primal.store_primal(context, value); }
    void load_derivative(IContext context, out DT value) { derivative.load_primal(context, value); }
    void store_derivative(IContext context, in DT value) { derivative.store_primal(context, value); }
}

typealias RORODiffPair<PT,DT> = BaseDiffPair<PT, DT, ValueRef<PT>, ValueRef<DT>>;
typealias RWRODiffPair<PT,DT> = BaseDiffPair<PT, DT, RWValueRef<PT>, ValueRef<DT>>;
typealias RORWDiffPair<PT,DT> = BaseDiffPair<PT, DT, ValueRef<PT>, RWValueRef<DT>>;
typealias RWRWDiffPair<PT,DT> = BaseDiffPair<PT, DT, RWValueRef<PT>, RWValueRef<DT>>;

struct TensorBuffer<T, let N : int> : IValueCallData<T>
{
    typealias primal_type = T;
    typealias derivative_type = T;

    RWStructuredBuffer<T> buffer;
    int[N] strides;

    T get(int[N] index) { return buffer[_idx(index, strides)]; }
    __subscript(int[N] index)->T { get { return get(index); } }

    void load_primal(IContext context, out primal_type value) { value = buffer[_idx(context, strides)]; }
    void store_primal(IContext context, in T value) {};
}

struct RWTensorBuffer<T, let N : int> : IValueCallData<T>
{
    typealias primal_type = T;
    typealias derivative_type = T;

    RWStructuredBuffer<T> buffer;
    int[N] strides;

    T get(int[N] index) { return buffer[_idx(index, strides)]; }
    void set(int[N] index, T value) { buffer[_idx(index, strides)] = value; }

    __subscript(int[N] index)->T { get { return get(index); } set { set(index, newValue); } }

    void load_primal(IContext context, out primal_type value) { value = buffer[_idx(context, strides)]; }
    void store_primal(IContext context, in primal_type value) { buffer[_idx(context, strides)] = value; }
}
