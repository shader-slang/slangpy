// SPDX-License-Identifier: Apache-2.0
// Original output from the SOA polynomial backwards pass

//------------------------------------------------------------------------------------------
// Helper types

int _idx<let N: int>(int[N] index, int[N] stride) {
    int idx = 0;
    for (int i = 0; i < N; i++) {
        idx += index[i] * stride[i];
    }
    return idx;
}

struct NDBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) {
        return buffer[_idx(index, strides)];
    }
    __subscript(int[N] index)->T
    {
        get { return get(index); }
    }
}

struct RWNDBuffer<T, let N : int> {
    RWStructuredBuffer<T> buffer;
    int[N] strides;
    T get(int[N] index) {
        return buffer[_idx(index, strides)];
    }
    void set(int[N] index, T value) {
        buffer[_idx(index, strides)] = value;
    }
    __subscript(int[N] index)->T
    {
        get { return get(index); }
        set { set(index, newValue); }
    }
}
//------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------
// User function
// import "e5d21fdc63252439";
[Differentiable]
float3 polynomial(float3 a, float3 b) {
    return a * a + b + 1;
}
//------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------
// Generated code

struct CallData
{
    int[1] _call_stride;
    int[1] _call_dim;
    uint3 _thread_count;
    NDBuffer<float,1> a__x_primal;
    NDBuffer<float,1> a__y_primal;
    NDBuffer<float,1> a__z_primal;
    NDBuffer<vector<float,3>,1> b_primal;
    RWNDBuffer<vector<float,3>,1> _result_primal;
}
ParameterBlock<CallData> call_data;

void load_a__x_primal(int[] call_id, out float val)
{
    val = call_data.a__x_primal[{call_id[0]}];
}
void load_a__y_primal(int[] call_id, out float val)
{
    val = call_data.a__y_primal[{call_id[0]}];
}
void load_a__z_primal(int[] call_id, out float val)
{
    val = call_data.a__z_primal[{call_id[0]}];
}
void load_a_primal(int[] call_id, out vector<float,3> val)
{
    load_a__x_primal(call_id, val.x);
    load_a__y_primal(call_id, val.y);
    load_a__z_primal(call_id, val.z);
}
void load_b_primal(int[] call_id, out vector<float,3> val)
{
    val = call_data.b_primal[{call_id[0]}];
}
void store__result_primal(int[] call_id, in vector<float,3> val)
{
    call_data._result_primal[{call_id[0]}] = val;
}

[Differentiable]
void _trampoline(out vector<float,3> _result, in vector<float,3> a, in vector<float,3> b)
{
    _result = polynomial(a, b);
}


[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    if (any(dispatchThreadID >= call_data._thread_count)) return;
    int[1] call_id;
    call_id[0] = (dispatchThreadID.x/call_data._call_stride[0]) % call_data._call_dim[0];
    vector<float,3> _result;
    vector<float,3> a;
    load_a_primal(call_id,a);
    vector<float,3> b;
    load_b_primal(call_id,b);
    _trampoline(_result, a, b);
    store__result_primal(call_id,_result);
}

