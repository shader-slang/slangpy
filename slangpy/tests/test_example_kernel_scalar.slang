// SPDX-License-Identifier: Apache-2.0


struct CallData_a
{
    typealias Result = float;
    Result primal;
    Result load() { return primal; }
}

struct CallData {
    RWStructuredBuffer<float> a;
    RWStructuredBuffer<float.Differential> a_grad;
    RWStructuredBuffer<float> b;
    RWStructuredBuffer<float.Differential> b_grad;
    RWStructuredBuffer<float> c;
    RWStructuredBuffer<float.Differential> c_grad;
};
CallData call_data;

static const uint3 TOTAL_THREADS = uint3(64, 1, 1);

[Differentiable]
void _trampoline( float32_t a,  float32_t b, out float32_t c) {
    user_func(a, b, c);
}

void get_a_index(uint3 dispatchThreadID, out int[] index) {
    index = {dispatchThreadID.x};
}
void get_b_index(uint3 dispatchThreadID, out int[] index) {
    index = { dispatchThreadID.x };
}
void get_c_index(uint3 dispatchThreadID, out int[] index) {
    index = { dispatchThreadID.x };
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID) {
    if (any(dispatchThreadID >= TOTAL_THREADS))
        return;

    int[1] a_index;
    int[1] b_index;
    int[1] c_index;

    get_a_index(dispatchThreadID, a_index);
    get_b_index(dispatchThreadID, b_index);
    get_c_index(dispatchThreadID, c_index);

    _trampoline(call_data.a[a_index[0]], call_data.b[b_index[0]], call_data.c[c_index[0]]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main_backwards(uint3 dispatchThreadID: SV_DispatchThreadID) {
    if (any(dispatchThreadID >= TOTAL_THREADS))
        return;

    int[1] a_index;
    int[1] b_index;
    int[1] c_index;

    get_a_index(dispatchThreadID, a_index);
    get_b_index(dispatchThreadID, b_index);
    get_c_index(dispatchThreadID, c_index);

    var a = diffPair(call_data.a[a_index[0]], call_data.a_grad[a_index[0]]);
    var b = diffPair(call_data.b[b_index[0]], call_data.b_grad[b_index[0]]);
    var c = call_data.c_grad[c_index[0]];

    bwd_diff(_trampoline)(a, b, c);

    call_data.a_grad[a_index[0]] = a.d;
    call_data.b_grad[b_index[0]] = b.d;
}
