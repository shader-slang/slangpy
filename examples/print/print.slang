// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl.device.print;

#if 0
struct MyType {
    int a = 1;
    int b = 2;
}

extension MyType : debug_print::IPrintable
{
    void print(inout debug_print::PrintOutput output)
    {
        output.format("MyType(a={},b={})", a, b);
    }
}
#endif

[shader("compute")]
[numthreads(2,2,1)]
void compute_main(uint2 tid: SV_DispatchThreadID)
{
    print("Hello from thread id {}", tid);
    print("matrix = {}", float4x4(0.5));
    // MyType mytype = {};
    // print("mytype = {}", mytype);
}
