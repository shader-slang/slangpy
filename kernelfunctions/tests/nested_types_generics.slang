import "slangpy";


struct Context : IContext
{
    uint3 thread_id;
    int[1] call_id;

    uint3 get_thread_id() { return thread_id; }
    int get_call_id(int dim) { return call_id[dim]; }
}

typealias _a__x = TensorBuffer<float, 1>;
typealias _a__y = TensorBuffer<float, 1>;
typealias _a__z = TensorBuffer<float, 1>;

struct _a
{
    typealias primal_type = float3;
    typealias derivative_type = float3;

    _a__x x;
    _a__y y;
    _a__z z;

    void load_primal(Context context, out primal_type value) {
        x.load_primal(context, value.x);
        y.load_primal(context, value.y);
        z.load_primal(context, value.z);
     }
}

typealias _b = TensorBuffer<float3, 1>;
typealias __result = RWTensorBuffer<float3, 1>;

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
    call_data.a.load_primal(context,a);
    _b::primal_type b;
    call_data.b.load_primal(context,b);
    __result::primal_type _result;
    _trampoline(a, b, _result);
    call_data._result.store_primal(context,_result);
}

