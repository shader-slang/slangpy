// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl.device.print;
import test_print_module2;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID)
{
    // print supports strings only
    print("Hello World!");

    // Test check if strings from modules linked via `import` are handled correctly.
    print_from_another_module(7);

    // print supports up to 8 arguments
    print("{}", 0);
    print("{}{}", 0, 1);
    print("{}{}{}", 0, 1, 2);
    print("{}{}{}{}", 0, 1, 2, 3);
    print("{}{}{}{}{}", 0, 1, 2, 3, 4);
    print("{}{}{}{}{}{}", 0, 1, 2, 3, 4, 5);
    print("{}{}{}{}{}{}{}", 0, 1, 2, 3, 4, 5, 6);
    print("{}{}{}{}{}{}{}{}", 0, 1, 2, 3, 4, 5, 6, 7);

    // print supports all basic types
    print("bool: false={}, true={}", false, true);
    // no int8 support on D3D12
#if !defined(__TARGET_D3D12__)
    print("int8: min={}, max={}, -123={}, 123={}", int8_t.minValue, int8_t.maxValue, int8_t(-123), int8_t(123));
#endif
    print(
        "int16: min={}, max={}, -12345={}, 12345={}",
        int16_t.minValue,
        int16_t.maxValue,
        int16_t(-12345),
        int16_t(12345)
    );
    print(
        "int32: min={}, max={}, -12345={}, 12345={}",
        int32_t.minValue,
        int32_t.maxValue,
        int32_t(-12345),
        int32_t(12345)
    );
    print(
        "int64: min={}, max={}, -12345={}, 12345={}",
        int64_t.minValue,
        int64_t.maxValue,
        int64_t(-12345),
        int64_t(12345)
    );
    // no uint8 support on D3D12
#if !defined(__TARGET_D3D12__)
    print("uint8: min={}, max={}, 0={}, 123={}", uint8_t.minValue, uint8_t.maxValue, uint8_t(0), uint8_t(123));
#endif
    print(
        "uint16: min={}, max={}, 12345={}, 23456={}",
        uint16_t.minValue,
        uint16_t.maxValue,
        uint16_t(12345),
        uint16_t(23456)
    );
    print(
        "uint32: min={}, max={}, 12345={}, 23456={}",
        uint32_t.minValue,
        uint32_t.maxValue,
        uint32_t(12345),
        uint32_t(23456)
    );
    print(
        "uint64: min={}, max={}, 12345={}, 23456={}",
        uint64_t.minValue,
        uint64_t.maxValue,
        uint64_t(12345),
        uint64_t(23456)
    );
    print(
        "float16: min={}, max={}, -123.45={:.5}, 123.45={:.5}",
        float16_t.minValue,
        float16_t.maxValue,
        float16_t(-123.45),
        float16_t(123.45)
    );
    print(
        "float32: min={}, max={}, -123.45={:.5}, 123.45={:.5}",
        float32_t.minValue,
        float32_t.maxValue,
        float32_t(-123.45),
        float32_t(123.45)
    );
    // no float64 support on Metal
#if !defined(__TARGET_METAL__)
    print(
        "float64: min={}, max={}, -123.45={:.5}, 123.45={:.5}",
        float64_t.minValue,
        float64_t.maxValue,
        float64_t(-123.45),
        float64_t(123.45)
    );
#endif

    // print supports vectors
    // no int8 support on D3D12 and Metal
#if !defined(__TARGET_D3D12__) && !defined(__TARGET_METAL__)
    print("int8_tX: {} {} {}", int8_t2(-4, -3), int8_t3(-2, -1, 0), int8_t4(1, 2, 3, 4));
#endif
    print("int16_tX: {} {} {}", int16_t2(-4, -3) * 1000, int16_t3(-2, -1, 0) * 1000, int16_t4(1, 2, 3, 4) * 1000);
    print(
        "int32_tX: {} {} {}",
        int32_t2(-4, -3) * 100000000l,
        int32_t3(-2, -1, 0) * 100000000l,
        int32_t4(1, 2, 3, 4) * 100000000l
    );
    print(
        "int64_tX: {} {} {}",
        int64_t2(-4, -3) * 10000000000000ll,
        int64_t3(-2, -1, 0) * 10000000000000ll,
        int64_t4(1, 2, 3, 4) * 10000000000000ll
    );
    // no uint8 support on D3D12 and Metal
#if !defined(__TARGET_D3D12__) && !defined(__TARGET_METAL__)
    print("uint8_tX: {} {} {}", uint8_t2(1, 2), uint8_t3(3, 4, 5), uint8_t4(6, 7, 8, 9));
#endif
    print("uint16_tX: {} {} {}", uint16_t2(1, 2) * 1000, uint16_t3(3, 4, 5) * 1000, uint16_t4(6, 7, 8, 9) * 1000);
    print(
        "uint32_tX: {} {} {}",
        uint32_t2(1, 2) * 100000000ul,
        uint32_t3(3, 4, 5) * 100000000ul,
        uint32_t4(6, 7, 8, 9) * 100000000ul
    );
    print(
        "uint64_tX: {} {} {}",
        uint64_t2(1, 2) * 10000000000000ull,
        uint64_t3(3, 4, 5) * 10000000000000ull,
        uint64_t4(6, 7, 8, 9) * 10000000000000ull
    );
    print(
        "float16_tX: {} {} {}",
        float16_t2(-4.h, -3.h) * 100.h,
        float16_t3(-2.h, -1.h, 0.h) * 100.h,
        float16_t4(1.h, 2.h, 3.h, 4.h) * 100.h
    );
    print(
        "float32_tX: {} {} {}",
        float32_t2(-4.f, -3.f) * 1e6f,
        float32_t3(-2.f, -1.f, 0.f) * 1e6f,
        float32_t4(1.f, 2.f, 3.f, 4.f) * 1e6f
    );
    // no float64 support on Metal
#if !defined(__TARGET_METAL__)
    print(
        "float64_tX: {} {} {}",
        float64_t2(1.0, 2.0) * 1e12,
        float64_t3(3.0, 4.0, 5.0) * 1e12,
        float64_t4(6.0, 7.0, 8.0, 9.0) * 1e12
    );
#endif

    // print supports matrices
    // no integer matrix support on Vulkan and Metal
#if !defined(__TARGET_VULKAN__) && !defined(__TARGET_METAL__)
    print("uint3x4: {}", uint3x4(uint4(1, 2, 3, 4), uint4(5, 6, 7, 8), uint4(9, 10, 11, 12)));
    print("uint4x4: {}", uint4x4(uint4(1, 2, 3, 4), uint4(5, 6, 7, 8), uint4(9, 10, 11, 12), uint4(13, 14, 15, 16)));
    print("uint4x3: {}", uint4x3(uint3(1, 2, 3), uint3(4, 5, 6), uint3(7, 8, 9), uint3(10, 11, 12)));
#endif

    // print supports format specifiers from fmtlib
    print("float3x3: {:+5.2f}", float3x3(float3(-4.0, -3.0, -1.0), float3(0.0, 1.0, 2.0), float3(3.0, 4.0, 5.0)));
}
