// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

// Interface for types which support atomic adds.
// "Atomic" is used loosely in the sense that memory will be consistent
// after a kernel completes, but intermediate states may not be. E.g.
// atomically adding a float4 will atomically add its components, and there
// is an observable state where e.g. x and y have been added but z and w have not yet.
// This definition is sufficient for our purposes (gradient accumulation),
// where the memory that is accumulated into is not read until the kernel completes
public interface IAtomicAddable
{
    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, This value);
}
public extension uint : IAtomicAddable
{
    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, uint value) { buf.InterlockedAdd(addr, value); }
}
public extension int64_t : IAtomicAddable
{
    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, int64_t value) { buf.InterlockedAddI64(addr, value); }
}
public extension float : IAtomicAddable
{
    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, float value) {
        buf.InterlockedAddF32(addr, value);

        // TODO: Atomic adds can be a performance issue, and there's a big benefit of local reductions like
        // the one below, where gradients are first accumulated across the warp and then atomically added only once.
        // Find a good spot for this (possibly moved into the AtomicTensor).
        // Although unlikely, this could be unsafe if threads in the warp simultaneously accumulate into the same
        // address into  different tensors. Need to double check whether this can happen before enabling this.
        // if (WaveActiveAllEqual(addr))
        // {
        //     float sum = WaveActiveSum(value);
        //     if (WaveIsFirstLane())
        //         buf.InterlockedAddF32(addr, sum);
        // }
        // else
        // {
        //     buf.InterlockedAddF32(addr, value);
        // }
    }
}
[__requiresNVAPI]
public extension half2 : IAtomicAddable
{
    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, half2 value) { buf._NvInterlockedAddFp16x2(addr, impl::asuint(value)); }
}
public extension half : IAtomicAddable
{
    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, half value)
    {
        half discardOut;
        buf.InterlockedAddF16(addr, value, discardOut);
    }
}

public extension<S : IAtomicAddable, T : ISizedArray<S, D>, let D : int> T : IAtomicAddable
{
    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, This value)
    {
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
        {
            S::atomicAdd(buf, addr + i * sizeof(S), value[i]);
        }
    }
}

// 2xfloat16 -> uint tricks
// This is lifted from Utils.Neural.TIN.TinCommon, where it was important for performance
// in half precision MLP training.
namespace impl
{
    __glsl_extension(GL_EXT_shader_explicit_arithmetic_types)
    uint asuint(vector<half, 2> a)
    {
        __target_switch
        {
#if 0 // Requires a custom version of DXC. Ignore for now
        case hlsl:
            __intrinsic_asm "asuint";
#endif
        case glsl:
            __intrinsic_asm "packFloat2x16";
        case spirv:
            return spirv_asm { result:$$uint = OpBitcast $a;};
        default:
            return (uint(asuint16(a.y)) << 16) | uint(asuint16(a.x));
        }
    }
}
