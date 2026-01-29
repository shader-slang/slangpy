// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl.device.print2;

#if 1
struct MyType {
    int a = 1;
    int b = 2;
    bool flag = false;
}

extension MyType : IPrintable
{
    void print(inout PrintOutput output)
    {
        output.format("MyType(a={},b={},flag={})", a, b, "TRUE");
        output.format(" ExtraData ");
    }
}
#endif

[shader("compute")]
[numthreads(2,2,1)]
void compute_main(uint2 tid: SV_DispatchThreadID)
{
    print("Hello from thread id {}", tid);
    print("matrix = {}", float4x4(0.5));
    MyType mytype = {};
    print("mytype = {} {}", mytype, 100);
    print("string: {}", "TestString");
    print("plain string!");
}
