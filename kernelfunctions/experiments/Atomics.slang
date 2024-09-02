import Helpers;

// Interface for types which support atomic adds.
// "Atomic" is used loosely in the sense that memory will be consistent
// after a kernel completes, but intermediate states may not be. E.g.
// atomically adding a float4 will atomically add its components, and there
// is an observable state where e.g. x and y have been added but z and w have not yet.
// This definition is sufficient for our purposes (gradient accumulation),
// where the memory that is accumulated into is not read until the kernel completes
interface IAtomicAddable
{
    static void atomicAdd(RWByteAddressBuffer buf, uint addr, This value);
}
// Helper interface for differential atomic types. This is to prove that differentials
// of atomically addable types are also atomically addable
interface IDiffAtomicAddable : IAtomicAddable, IDifferentiable
{
    associatedtype Differential : IDiffAtomicAddable;
}

extension uint : IAtomicAddable
{
    static void atomicAdd(RWByteAddressBuffer buf, uint addr, uint value) { buf.InterlockedAdd(addr, value); }
}
extension int64_t : IAtomicAddable
{
    static void atomicAdd(RWByteAddressBuffer buf, uint addr, int64_t value) { buf.InterlockedAddI64(addr, value); }
}
extension float : IDiffAtomicAddable
{
    static void atomicAdd(RWByteAddressBuffer buf, uint addr, float value) { buf.InterlockedAddF32(addr, value); }
}
[__requiresNVAPI]
extension half2 : IDiffAtomicAddable
{
    static void atomicAdd(RWByteAddressBuffer buf, uint addr, half2 value) { buf._NvInterlockedAddFp16x2(addr, impl::asuint(value)); }
}
extension half : IDiffAtomicAddable
{
    static void atomicAdd(RWByteAddressBuffer buf, uint addr, half value)
    {
        half discardOut;
        buf.InterlockedAddF16(addr, value, discardOut);
    }
}

__generic<S : IAtomicAddable, T : ISizedArray<S, D>, let D : int>
extension T : IAtomicAddable
{
    static void atomicAdd(RWByteAddressBuffer buf, uint addr, This value)
    {
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
        {
            S::atomicAdd(buf, addr + i * sizeof(T), value[i]);
        }
    }
}

// Prove that Array, vector and matrix conform to IDiffAtomicAddable if their elements do
__generic<T : IDiffAtomicAddable, let D : int>
extension Array<T, D> : IDiffAtomicAddable
{
    typealias Differential = Array<T.Differential, D>;
}
__generic<T : IDiffAtomicAddable, let D : int>
extension vector<T, D> : IDiffAtomicAddable
{
    typealias Differential = vector<T.Differential, D>;
}
__generic<T : IDiffAtomicAddable, let R : int, let C : int, let L : int>
extension matrix<T, R, C, L> : IDiffAtomicAddable
{
    typealias Differential = matrix<T.Differential, R, C, L>;
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
