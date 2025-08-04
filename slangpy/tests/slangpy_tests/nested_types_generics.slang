// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import "slangpy";

typealias _a__x = NDBuffer<float, 1>;
typealias _a__y = NDBuffer<float, 1>;
typealias _a__z = NDBuffer<float, 1>;

struct _a
{
    typealias primal_type = float3;
    typealias derivative_type = float3;

    _a__x x;
    _a__y y;
    _a__z z;

    void load(Context1D context, out primal_type value) {
        x.load(context, value.x);
        y.load(context, value.y);
        z.load(context, value.z);
     }
}

typealias _b = NDBuffer<float3, 1>;
typealias __result = RWNDBuffer<float3, 1>;

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
void compute_main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    if (any(dispatchThreadID >= call_data._thread_count)) return;
    int[1] call_id = {
        (dispatchThreadID.x / call_data._call_stride[0]) % call_data._call_dim[0],
    };
    Context1D context = { dispatchThreadID, call_id };
    float3 a;
    call_data.a.load(context, a);
    float3 b;
    call_data.b.load(context, b);
    float3 _result;
    _trampoline(a, b, _result);
    call_data._result.store(context,_result);
}
