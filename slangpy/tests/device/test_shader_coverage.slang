// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

int classify(int x)
{
    if (x < 0)
        return -x;
    else
        return 2 * x;
}
