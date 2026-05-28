// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl.device.print;

[shader("compute")]
[numthreads(2,2,1)]
void compute_main(uint2 tid: SV_DispatchThreadID)
{
    print("Hello from thread id {}", tid);
    print("matrix = {}", float4x4(0.5));
}
