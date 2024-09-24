/*
Idx  Name       Input Type                 Output Type          Input Shape  Argument Shape  Type Shape  Transform      
0    index      RWTensorBuffer<int,1>      vector<int,2>        [?]          []              [2]         [1]            
1    texture    RWTensorBuffer<float,3>    NDBuffer<float,2>    [?, ?, ?]    [?]             [?, ?]      [0, 1, 2]      
2    _result    RWTensorBuffer<float,1>    float                [?]          ?               []          [0]            
*/


import "slangpy";

struct Context: IContext
{
    uint3 thread_id;
    int[1] call_id;
    uint3 get_thread_id() { return thread_id; }
    int get_call_id(int dim) { return call_id[dim]; }
}

typealias _index = NDBuffer<int,1>;
typealias _texture = NDBuffer<float,3>;

struct __result
{
    RWTensorBuffer<float,1> primal;
    NoneType derivative;
    void load_primal(IContext context, out float value) { primal.load_primal(context, value); }
    void store_primal(IContext context, in float value) { primal.store_primal(context, value); }
    void load_derivative(IContext context, out float value) { derivative.load_primal(context, value); }
    void store_derivative(IContext context, in float value) { derivative.store_primal(context, value); }
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
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    if (any(dispatchThreadID >= call_data._thread_count)) return;
    Context context;
    context.thread_id = dispatchThreadID;
    context.call_id[0] = (dispatchThreadID.x/call_data._call_stride[0]) % call_data._call_dim[0];
    vector<int,2> index;
    call_data.index.load_primal(context,index);
    NDBuffer<float,2> texture;
    call_data.texture.load_primal(context,texture);
    float _result;
    _trampoline(index, texture, _result);
    call_data._result.store_primal(context,_result);
}

