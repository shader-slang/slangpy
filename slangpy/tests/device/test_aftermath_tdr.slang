// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(RWStructuredBuffer<uint> buffer)
{
    uint count = 0;
    while (buffer[1] != 0xaaaaaaaa)
    {
        ++count;
        if (count % 1000 == 0)
            buffer[0] = count;
    }
}
