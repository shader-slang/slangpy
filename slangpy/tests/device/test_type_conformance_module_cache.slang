// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import test_type_conformance;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main_2(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<uint> result)
{
    uint i = tid.x;
    uint type_id = tid.x;
    uint dummy = 0;
    IFoo foo = createDynamicObject<IFoo>(type_id, dummy);
    result[i] = foo.get();
}
