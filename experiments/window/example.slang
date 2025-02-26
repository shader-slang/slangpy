// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


float4 sinwave(float t) {
    float3 sinwave = sin(t+float3(0,1,2))*0.5+0.5;
    return float4(sinwave, 1);
}
