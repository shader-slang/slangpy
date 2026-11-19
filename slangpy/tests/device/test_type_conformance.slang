// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

public interface IFoo {
    uint get();
}

public struct Foo1 : IFoo {
    public uint dummy;
    public uint get() { return 1; }
}

public struct Foo2 : IFoo {
    public uint dummy;
    public uint get() { return 2; }
}

public struct Foo3 : IFoo {
    public uint dummy;
    public uint get() { return 3; }
}

public struct Foo4 : IFoo {
    public uint dummy;
    public uint get() { return 4; }
}

[shader("compute")]
[numthreads(1, 1, 1)]
public void compute_main(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<uint> result)
{
    uint i = tid.x;
    uint type_id = tid.x;
    uint dummy = 0;
    IFoo foo = createDynamicObject<IFoo>(type_id, dummy);
    result[i] = foo.get();
}
