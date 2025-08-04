// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module test_print_module2;
import sgl.device.print;

public void print_from_another_module(int arg)
{
    print("This is Print from Another Module: {}", arg);
}
