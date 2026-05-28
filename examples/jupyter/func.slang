// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[Differentiable]
float f(float x, no_diff float A, no_diff float B, no_diff float C)
{
    return A * sin(x * B) + C;
}


[shader("compute")]
[numthreads(32, 1, 1)]
void main(
    uint tid: SV_DispatchThreadID,
    StructuredBuffer<float> buf_x,
    RWStructuredBuffer<float> buf_y,
    RWStructuredBuffer<float> buf_dydx,
    uniform float A,
    uniform float B,
    uniform float C
)
{
    float x = buf_x[tid];
    float y = f(x, A, B, C);
    let dxdy = fwd_diff(f)(diffPair(x, 1.0), A, B, C);
    buf_y[tid] = y;
    buf_dydx[tid] = dxdy.d;
}
