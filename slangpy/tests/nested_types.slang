// SPDX-License-Identifier: Apache-2.0
int _idx<let N : int>(int[N] index, int[N] stride) {
    int idx = 0;
    for (int i = 0; i < N; i++) { idx += index[i] * stride[i]; }
    return idx;
}
struct NDBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) { return buffer[_idx(index, strides)]; }
    __subscript(int[N] index)->T { get { return get(index); } }
}
struct RWNDBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) { return buffer[_idx(index, strides)]; }
    void set(int[N] index, T value) { buffer[_idx(index, strides)] = value; }
    __subscript(int[N] index)->T { get { return get(index); } set { set(index, newValue); } }
}

struct Context
{
    uint3 thread_id;
    int[1] call_id;
}

struct _a__x
{
    typealias primal_type = float;
    typealias derivative_type = float;
    NDBuffer<primal_type, 1> primal;
    void load(Context context, out primal_type value) { value = this.primal[ { context.call_id[0] }]; }
}
struct _a__y
{
    typealias primal_type = float;
    typealias derivative_type = float;
    NDBuffer<primal_type, 1> primal;
    void load(Context context, out primal_type value) { value = this.primal[ { context.call_id[0] }]; }
}
struct _a__z
{
    typealias primal_type = float;
    typealias derivative_type = float;
    NDBuffer<primal_type, 1> primal;
    void load(Context context, out primal_type value) { value = this.primal[ { context.call_id[0] }]; }
}

struct _a
{
    typealias primal_type = float3;
    typealias derivative_type = float3;

    _a__x x;
    _a__y y;
    _a__z z;

    void load(Context context, out primal_type value) {
        x.load(context, value.x);
        y.load(context, value.y);
        z.load(context, value.z);
     }
}

struct _b
{
    typealias primal_type = float3;
    typealias derivative_type = float3;
    NDBuffer<primal_type,1> primal;
    void load(Context context, out primal_type value) { value = this.primal[{context.call_id[0]}]; }
}

struct __result
{
    typealias primal_type = float3;
    typealias derivative_type = float3;
    RWNDBuffer<primal_type,1> primal;
    void load(Context context, out primal_type value) { value = this.primal[{context.call_id[0]}]; }
    void store(Context context, in primal_type value) { this.primal.set({context.call_id[0]}, value); }
}

struct CallData
{
    int[1] _call_stride;
    int[1] _call_dim;
    uint3 _thread_count;
    _a a;
    _b b;
    __result _result;
}
ParameterBlock<CallData> call_data;

[Differentiable]
float3 polynomial(float3 a, float3 b)
{
    return a * a + b + 1;
}

[Differentiable]
void _trampoline(in float3 a, in float3 b, no_diff out float3 _result)
{
    _result = polynomial(a, b);
}


[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    if (any(dispatchThreadID >= call_data._thread_count)) return;
    Context context;
    context.thread_id = dispatchThreadID;
    context.call_id[0] = (dispatchThreadID.x/call_data._call_stride[0]) % call_data._call_dim[0];
    _a::primal_type a;
    call_data.a.load(context,a);
    _b::primal_type b;
    call_data.b.load(context,b);
    __result::primal_type _result;
    _trampoline(a, b, _result);
    call_data._result.store(context,_result);
}
