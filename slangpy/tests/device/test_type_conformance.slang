// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

interface IFoo {
    uint get();
}

struct Foo1 : IFoo {
    uint dummy;
    uint get() { return 1; }
}

struct Foo2 : IFoo {
    uint dummy;
    uint get() { return 2; }
}

struct Foo3 : IFoo {
    uint dummy;
    uint get() { return 3; }
}

struct Foo4 : IFoo {
    uint dummy;
    uint get() { return 4; }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<uint> result)
{
    uint i = tid.x;
    uint type_id = tid.x;
    uint dummy = 0;
    IFoo foo = createDynamicObject<IFoo>(type_id, dummy);
    result[i] = foo.get();
}
