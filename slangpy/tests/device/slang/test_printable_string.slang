// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sgl.device.print;
RWStructuredBuffer<uint> result;

PrintableString string_from_function() { return "string_from_function"; }

struct Struct1
{
    __init() { }
    PrintableString string_from_method() { return "string_from_method"; }
}

struct Struct2
{
    __init() { }
    static const PrintableString string_from_static_const = "string_from_static_const";
}

interface Interface
{
    static const PrintableString string_from_interface_static_const;
    PrintableString string_from_interface_method();
}

struct Struct3 : Interface
{
    __init() { }

    static const PrintableString string_from_interface_static_const = "string_from_interface_static_const";
    PrintableString string_from_interface_method() { return "string_from_interface_method"; }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main()
{
    Struct1 struct1 = {};
    Struct2 struct2 = {};
    Struct3 struct3 = {};

    result[0] = string_from_function().hash;
    result[1] = struct1.string_from_method().hash;
    // Disabled due to: https://github.com/shader-slang/slang/issues/8694
    // result[2] = struct2.string_from_static_const.hash;
    // result[3] = struct3.string_from_interface_static_const.hash;
    result[4] = struct3.string_from_interface_method().hash;
}
