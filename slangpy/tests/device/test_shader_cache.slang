// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl.device.print;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main()
{
    print("Hello shader cache!");
}
