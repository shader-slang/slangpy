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
    public static void atomicAdd(This* buf, uint element, This value);
}

// Automatically implement the ptr version of atomicAdd for types that implement IArithmeticAtomicable.
public extension<T: IArithmeticAtomicable> T
{
    public static void atomicAdd(This* buf, uint element, This value)
    {
        InterlockedAdd(buf[element], value);
    }
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

// #pragma warning(push)
// #pragma warning(disable: 30856)

// Add atomic support for per element operations on arrays or vectors of atomics.
public extension<S : IAtomicAddable, T : ISizedArray<S, D>, let D : int> T : IAtomicAddable
{
    // Byte address buffer version increments each element by stepping through the buffer
    public static void atomicAdd(RWByteAddressBuffer buf, uint addr, This value)
    {
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
        {
            S::atomicAdd(buf, addr + i * sizeof(S), value[i]);
        }
    }

    // Ptr version casts the T* (eg float3*) to the element type, S* (eg float*) and then
    // increments each sub element individually.
    public static void atomicAdd(T* buf, uint element, This value)
    {
        S* buf2 = (S*)buf;
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
        {
            S::atomicAdd(buf2, element*D+i, value[i]);
        }
    }
}

// #pragma warning(pop)

// Extension of the storage traits object in slangpy/slang/core.slang that defines how to
// store a buffer of atomics, and where appropriate uses IAtomicAddable to define the atomic add operation.
public extension<T: IAtomicAddable> StorageTraits<T>
{
#ifdef __TARGET_CUDA__
    public typealias AtomicBufferType = T*;
    public static void atomicAdd(AtomicBufferType buffer, uint idx, T value) { T::atomicAdd(buffer, idx, value); }
    public static T atomicLoad(AtomicBufferType buffer, uint idx) { return buffer[idx]; }
#else
    public typealias AtomicBufferType = RWByteAddressBuffer;
    public static void atomicAdd(AtomicBufferType buffer, uint idx, T value) { T::atomicAdd(buffer, idx*sizeof(T), value); }
    public static T atomicLoad(AtomicBufferType buffer, uint idx) { return buffer.Load<T>(idx*sizeof(T));}
#endif
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
