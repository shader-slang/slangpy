// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/*
Idx  Name       Input Type                 Output Type          Input Shape  Argument Shape  Type Shape  Transform
0    index      RWNDBuffer<int,1>      vector<int,2>        [?]          []              [2]         [1]
1    texture    RWNDBuffer<float,3>    NDBuffer<float,2>    [?, ?, ?]    [?]             [?, ?]      [0, 1, 2]
2    _result    RWNDBuffer<float,1>    float                [?]          ?               []          [0]
*/


import "slangpy";

typealias Context = ContextND<1>;

typealias _index = NDBuffer<int,1>;
typealias _texture = NDBuffer<float,3>;

struct __result
{
    RWNDBuffer<float,1> primal;
    NoneType derivative;
    [Differentiable, BackwardDerivative(load_bwd)]
    void load(Context1D context, out float value) { primal.load(context, value); }
    void load_bwd(Context1D context, in float value) { derivative.store(context, value); }

    [Differentiable, BackwardDerivative(store_bwd)]
    void store(Context1D context, in float value) { primal.store(context, value); }
    void store_bwd(Context1D context, inout DifferentialPair<float> value) {
        float grad;
        derivative.load(context, grad);
        value = diffPair(value.p, grad);
    }
}

struct CallData
{
    int[1] _call_stride;
    int[1] _call_dim;
    uint3 _thread_count;
    _index index;
    _texture texture;
    __result _result;
}
ParameterBlock<CallData> call_data;

float _read_slice(int2 index, NDBuffer<float, 2> texture)
{
    return texture[ { index.x, index.y }];
}

[Differentiable]
void _trampoline(no_diff in vector<int,2> index, no_diff in NDBuffer<float,2> texture, no_diff out float _result)
{
    _result = _read_slice(index, texture);
}


[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    if (any(dispatchThreadID >= call_data._thread_count)) return;
    int[1] call_id = {
        (dispatchThreadID.x / call_data._call_stride[0]) % call_data._call_dim[0],
    };
    Context1D context = { dispatchThreadID, call_id };
    vector<int,2> index;
    call_data.index.load(context.map(0),index);
    NDBuffer<float,2> texture;
    call_data.texture.load(context,texture);
    float _result;
    _trampoline(index, texture, _result);
    call_data._result.store(context,_result);
}
