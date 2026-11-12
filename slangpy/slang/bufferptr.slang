// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module bufferptr;

namespace buffer_ptr_detail {
[ForceInline]
void validate_element<T>()
{
    static_assert(sizeof(T) > 0, "Element access requires a non-void type");
}

[ForceInline]
void validate_access<T>()
{
    static_assert(sizeof(T) > 0 && sizeof(T) % 4 == 0, "Portable access size must be divisible by four");
}

static const uint64_t NULL_BITS = 0xffffffffffffffffull;

[ForceInline]
void validate<Access access, let load_alignment : uint>()
{
    static_assert(access == Access.Read || access == Access.ReadWrite, "Unsupported BufferPtr access");
    static_assert(
        load_alignment == 0 || load_alignment == 4 || load_alignment == 8 || load_alignment == 16,
        "BufferPtr load_alignment must be 0 (plain access), 4, 8 or 16"
    );
}

[ForceInline]
void validate_equality<T, U, Access access, Access other_access>()
{
    // Reject inside the selected overload: excluding it would allow bool fallback.
    static_assert(
        __type_equals<T, U>() || __type_equals<T, void>() || __type_equals<U, void>(),
        "BufferPtr equality requires matching pointee types or a void pointee"
    );
#if defined(__TARGET_D3D12__)
    static_assert(access == other_access, "D3D12 BufferPtr equality requires matching SRV/UAV access");
#endif
}

}

/// A non-owning GPU pointer for typed indexing and byte-oriented buffer access.
/// Use p.as<T>()[index] for elements, or p.load<T>(byte_offset) for raw access.
/// @param T The element type for subscripting and arithmetic; void provides raw access without element indexing.
/// @param access Read-only by default; use RWBufferPtr for read-write access. Read does not imply immutable data.
/// @param load_alignment Zero uses plain loads/stores. A nonzero value supplies an explicit alignment operand
/// for loads/stores; accesses support 4, 8 or 16 bytes. The caller must satisfy it at every accessed address.
/// @remarks The caller retains the resource and provides synchronization. Shader operations do not check
/// lifetime, bounds or address overflow. Memory access requires a complete value with a matching backend
/// layout, a nonzero size divisible by four, and an address aligned to at least four bytes and the type's
/// natural alignment. Plain access preserves these requirements; load_alignment does not change layout or stride.
/// D3D12 nested records with implicit tail padding require verified raw/structured layout compatibility;
/// matching total stride is insufficient. See https://github.com/shader-slang/slang/issues/14311.
/// Supports Vulkan, CUDA, D3D12 and Metal. Metal requires device-wide residency sets.
/// Import bufferptr to use this module independently of the functional SlangPy library.
public struct BufferPtr<T = void, Access access = Access.Read, let load_alignment : uint = 0>
{
#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__) || defined(__TARGET_METAL__)
    private Ptr<uint8_t, access> data = Ptr<uint8_t, access>(buffer_ptr_detail::NULL_BITS);
#elif defined(__TARGET_D3D12__)
    // Low word: descriptor; high word: byte_offset. uint64 preserves 8-byte ABI alignment.
    private uint64_t data = buffer_ptr_detail::NULL_BITS;
#else
#error BufferPtr supports D3D12, Vulkan, CUDA and Metal.
#endif

    // Construction and casts.

    /// Construct from backend-specific bits with matching access encoding. All-ones is null;
    /// zero-filled storage is not null on D3D12 because descriptor zero is valid.
    public __init(uint64_t bits)
    {
        buffer_ptr_detail::validate<access, load_alignment>();
#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__) || defined(__TARGET_METAL__)
        data = Ptr<uint8_t, access>(bits);
#else
        data = bits;
#endif
    }

    /// Construct a null pointer.
    public __init() { this = BufferPtr<T, access, load_alignment>(buffer_ptr_detail::NULL_BITS); }

    // Slang's NullPtrToPtr conversion cost.
    __implicit_conversion(150) [ForceInline]
    public __init(NullPtr value)
    {
        this = BufferPtr<T, access, load_alignment>();
    }

    /// Reinterpret the same location using the destination's element type and load_alignment.
    /// Vulkan/CUDA allow access casts like Ptr; casting to writable requires writable memory.
    /// D3D12 requires matching access because SRV/UAV descriptors differ.
    public __init<U, Access other_access, let other_load_alignment : uint>(
        BufferPtr<U, other_access, other_load_alignment> value
    )
    {
#if defined(__TARGET_D3D12__)
        // Keep this overload for both access modes: otherwise Slang can route a
        // forbidden explicit cast through the integer constructor.
        static_assert(access == other_access, "D3D12 BufferPtr access casts require a different SRV/UAV descriptor");
#endif
        this = BufferPtr<T, access, load_alignment>(value.bits());
    }

    // Pointer inspection and views.

    /// Return transport bits; these are not a device address on D3D12.
    [ForceInline]
    public uint64_t bits()
    {
        buffer_ptr_detail::validate<access, load_alignment>();
        return (uint64_t)data;
    }

    /// Test for the null sentinel only; does not check lifetime, bounds or alignment.
    public bool is_valid() { return bits() != buffer_ptr_detail::NULL_BITS; }

    /// Reinterpret this location as U, preserving access and load_alignment.
    /// The caller must ensure the stored data has the layout required by U.
    public BufferPtr<U, access, load_alignment> as<U>() { return BufferPtr<U, access, load_alignment>(bits()); }

    /// Select the default alignment operand for subsequent loads and stores; zero restores plain access.
    /// This does not check, round or move the address.
    public BufferPtr<T, access, new_load_alignment> with_load_alignment<let new_load_alignment : uint>()
    {
        return BufferPtr<T, access, new_load_alignment>(bits());
    }

    // Address manipulation.

    /// Advance by a signed or unsigned byte offset, preserving load_alignment.
    /// Requires a non-null pointer and a result within the same allocation or one-past-end,
    /// without address overflow (or overflow of D3D12's 32-bit byte offset).
    [ForceInline]
    public BufferPtr<T, access, load_alignment> byte_offset<I : __BuiltinIntegerType>(I byte_offset)
    {
#if defined(__TARGET_D3D12__)
        return BufferPtr<T, access, load_alignment>(
            uint64_t(uint(data)) | (uint64_t(uint(data >> 32) + byte_offset.toUInt()) << 32)
        );
#else
        return BufferPtr<T, access, load_alignment>(bits() + byte_offset.toUInt64());
#endif
    }

    /// Advance by count elements using the backend's element stride, including tail padding.
    /// Requires non-void T and the same bounds/overflow conditions as byte_offset().
    /// Preserves load_alignment. Indexing and ++/-- use this same stride.
    public BufferPtr<T, access, load_alignment> element_offset<I : __BuiltinIntegerType>(I count)
    {
        buffer_ptr_detail::validate_element<T>();
#if defined(__TARGET_CUDA__) || defined(__TARGET_METAL__)
        // CUDA and Metal's native layout can add padding inside aggregates as well as at the end.
        return BufferPtr<T, access, load_alignment>(uint64_t(Ptr<T, access>(bits()) + int64_t(count.toUInt64())));
#else
        // Vulkan/D3D use the default buffer layout. Keep address arithmetic explicit:
        // native Vulkan access chains can miscompile negative offsets (Slang #14305).
        // https://github.com/shader-slang/slang/issues/14305
        let size = uint64_t(sizeof(T, DefaultDataLayout));
        let alignment = uint64_t(alignof(T, DefaultDataLayout));
        let stride = (size + alignment - 1) & ~(alignment - 1);
        return byte_offset(count.toUInt64() * stride);
#endif
    }

    // Byte and element loads.

    /// Load U with an explicit alignment operand, overriding the type's load_alignment.
    /// @param byte_offset Nonnegative byte displacement from this pointer, including on D3D12.
    /// @param operation_alignment Alignment in bytes (4, 8 or 16) promised at the effective address.
    [ForceInline]
    public U load_aligned<U, let operation_alignment : uint>(uint64_t byte_offset = 0)
    {
        buffer_ptr_detail::validate<access, load_alignment>();
        buffer_ptr_detail::validate<access, operation_alignment>();
        static_assert(operation_alignment >= 4, "Portable buffer access requires at least four-byte load_alignment");
        buffer_ptr_detail::validate_access<U>();
#if defined(__TARGET_D3D12__)
        // The descriptor selects the buffer; the high word supplies its base byte offset.
        let effective_byte_offset = uint(data >> 32) + uint(byte_offset);
        if (access == Access.Read) {
            DescriptorHandle<ByteAddressBuffer> handle = { uint2(uint(data), 0) };
            return handle.LoadAligned<U>(effective_byte_offset, operation_alignment);
        } else {
            DescriptorHandle<RWByteAddressBuffer> handle = { uint2(uint(data), 0) };
            return handle.LoadAligned<U>(effective_byte_offset, operation_alignment);
        }
#else
        return loadAligned<operation_alignment>((Ptr<U>)(bits() + uint64_t(byte_offset)));
#endif
    }

    /// Load a complete, readable U at a byte offset relative to this pointer.
    /// Uses plain access by default; a nonzero load_alignment opts into aligned access.
    /// @param byte_offset Nonnegative byte displacement; use byte_offset() to move backwards first.
    [ForceInline]
    public U load<U>(uint64_t byte_offset = 0)
    {
        buffer_ptr_detail::validate<access, load_alignment>();
        buffer_ptr_detail::validate_access<U>();
        if (load_alignment != 0)
            return load_aligned<U, load_alignment>(byte_offset);
#if defined(__TARGET_D3D12__)
        let effective_byte_offset = uint(data >> 32) + uint(byte_offset);
        if (access == Access.Read) {
            DescriptorHandle<ByteAddressBuffer> handle = { uint2(uint(data), 0) };
            return handle.Load<U>(effective_byte_offset);
        } else {
            DescriptorHandle<RWByteAddressBuffer> handle = { uint2(uint(data), 0) };
            return handle.Load<U>(effective_byte_offset);
        }
#else
        return Ptr<U, access>(bits() + byte_offset)[0];
#endif
    }
}

public extension<T, Access access, let load_alignment : uint> BufferPtr<T, access, load_alignment>
{
    /// Read an element by value.
    /// @param index Element displacement from this pointer, using the same stride as element_offset().
    /// @remarks T must be non-void. The selected element must be readable and satisfy the shared
    /// memory-access requirements and any explicit load_alignment. A one-past-end pointer cannot be read.
    public __subscript<I : __BuiltinIntegerType>(I index)->T
    {
        [ForceInline]
        get
        {
            return element_offset(index).load<T>();
        }
    }
}

/// Writable buffer access. Stores are ordinary non-atomic memory operations.
/// D3D12 double stores are unsupported: Slang can silently corrupt constant values.
/// See https://github.com/shader-slang/slang/issues/14259. This restriction is not enforced by the type system.
public extension<T, let load_alignment : uint> BufferPtr<T, Access.ReadWrite, load_alignment>
{
    // Byte and element stores.

    /// Store a complete U with an explicit alignment operand, overriding load_alignment.
    /// @param byte_offset Nonnegative byte displacement from this pointer, including on D3D12.
    /// @param store_alignment Alignment in bytes (4, 8 or 16) promised at the effective address.
    [ForceInline]
    public void store_aligned<U, let store_alignment : uint>(uint64_t byte_offset, U value)
    {
        buffer_ptr_detail::validate<Access.ReadWrite, load_alignment>();
        buffer_ptr_detail::validate<Access.ReadWrite, store_alignment>();
        static_assert(store_alignment >= 4, "Portable buffer access requires at least four-byte load_alignment");
        buffer_ptr_detail::validate_access<U>();
#if defined(__TARGET_D3D12__)
        DescriptorHandle<RWByteAddressBuffer> handle = { uint2(uint(bits()), 0) };
        let effective_byte_offset = uint(bits() >> 32) + uint(byte_offset);
        handle.Store<U>(effective_byte_offset, value, store_alignment);
#else
        storeAligned<store_alignment>((Ptr<U>)(bits() + uint64_t(byte_offset)), value);
#endif
    }

    /// Store U at this pointer's address, using the type's load_alignment.
    public void store<U>(U value)
    {
        store<U>(0, value);
    }

    /// Store a complete U at a byte offset relative to this pointer within the writable allocation.
    /// Uses plain access by default; a nonzero load_alignment opts into aligned access.
    /// @param byte_offset Nonnegative byte displacement; use byte_offset() to move backwards first.
    [ForceInline]
    public void store<U>(uint64_t byte_offset, U value)
    {
        buffer_ptr_detail::validate<Access.ReadWrite, load_alignment>();
        buffer_ptr_detail::validate_access<U>();
        if (load_alignment != 0) {
            store_aligned<U, load_alignment>(byte_offset, value);
            return;
        }
#if defined(__TARGET_D3D12__)
        DescriptorHandle<RWByteAddressBuffer> handle = { uint2(uint(bits()), 0) };
        let effective_byte_offset = uint(bits() >> 32) + uint(byte_offset);
        handle.Store<U>(effective_byte_offset, value);
#else
        Ptr<U>(bits() + byte_offset)[0] = value;
#endif
    }

    /// Read or write an element by value; this does not return a native reference.
    /// @param index Element displacement from this pointer, using the same stride as element_offset().
    /// @remarks The readable subscript's type, bounds and alignment requirements also apply here;
    /// writes require a complete writable element. Field updates can read and write the entire T,
    /// so concurrent writes to different fields of one element are not independent or atomic.
    public __subscript<I : __BuiltinIntegerType>(I index)->T
    {
        [ForceInline]
        get
        {
            return element_offset(index).load<T>();
        }
        [ForceInline]
        [nonmutating]
        set
        {
            element_offset(index).store<T>(newValue);
        }
    }
}

/// Writable counterpart to the read-only BufferPtr default.
public typealias RWBufferPtr<T = void, let load_alignment : uint = 0> = BufferPtr<T, Access.ReadWrite, load_alignment>;

// Aligned access. Preserve Slang's spellings for these overloads.

/// Load a value with an explicit alignment operand, overriding p's load_alignment.
/// @param operation_alignment Alignment promised by the caller at p's address.
/// @param p Pointer to a complete readable T, satisfying BufferPtr's memory-access requirements.
[ForceInline]
public T loadAligned<let operation_alignment : int, T, Access access, let load_alignment : uint>(
    BufferPtr<T, access, load_alignment> p
)
{
    return p.load_aligned<T, uint(operation_alignment)>();
}

/// Store a value with an explicit alignment operand, overriding p's load_alignment.
/// @param operation_alignment Alignment promised by the caller at p's address.
/// @param p Pointer to a complete writable T, satisfying BufferPtr's memory-access requirements.
/// @param value The value to store. The operation is not atomic.
[ForceInline]
public void storeAligned<let operation_alignment : int, T, let load_alignment : uint>(
    BufferPtr<T, Access.ReadWrite, load_alignment> p,
    T value
)
{
    p.store_aligned<T, uint(operation_alignment)>(0, value);
}

// Comparisons and conversions.

/// Compare matching pointees or typed/void pairs, independently of load_alignment.
/// Numeric operands are rejected; use nullptr for null checks or bits() for encoded-value comparisons.
/// Booleans and other bool-convertible operands can still fall back to comparing boolean values; only compare
/// against compatible BufferPtr values or nullptr. See https://github.com/shader-slang/slang/issues/14268.
/// D3D12 requires matching access and compares encoded view identity; separate descriptors
/// for the same allocation are not canonicalized. Native backends may compare across access modes.
public bool
operator ==<T, U, Access access, Access other_access, let load_alignment : uint, let other_load_alignment : uint>(
    BufferPtr<T, access, load_alignment> a,
    BufferPtr<U, other_access, other_load_alignment> b
)
{
    buffer_ptr_detail::validate_equality<T, U, access, other_access>();
    return a.bits() == b.bits();
}

public bool
operator !=<T, U, Access access, Access other_access, let load_alignment : uint, let other_load_alignment : uint>(
    BufferPtr<T, access, load_alignment> a,
    BufferPtr<U, other_access, other_load_alignment> b
)
{
    return !(a == b);
}

// Numeric operands must not reach the implicit bool comparison. Compare bits()
// explicitly for transport values, or use nullptr to test for the null sentinel.
// Slang fallback: https://github.com/shader-slang/slang/issues/14268#issuecomment-6636216092
public bool operator ==<T, Access access, let load_alignment : uint, N : __BuiltinArithmeticType>(
    BufferPtr<T, access, load_alignment> p,
    N value
)
{
    static_assert(false, "BufferPtr cannot be compared with a number; use nullptr or compare bits() explicitly");
    return false;
}

public bool operator ==<T, Access access, let load_alignment : uint, N : __BuiltinArithmeticType>(
    N value,
    BufferPtr<T, access, load_alignment> p
)
{
    return p == value;
}

public bool operator !=<T, Access access, let load_alignment : uint, N : __BuiltinArithmeticType>(
    BufferPtr<T, access, load_alignment> p,
    N value
)
{
    return !(p == value);
}

public bool operator !=<T, Access access, let load_alignment : uint, N : __BuiltinArithmeticType>(
    N value,
    BufferPtr<T, access, load_alignment> p
)
{
    return !(p == value);
}

/// Compare positions within the same allocation (and the same descriptor view on D3D12).
/// Requires matching pointee/access types. Descriptor order is not address order.
public __generic<T, Access access, let load_alignment : uint, let other_load_alignment : uint> bool
operator<(BufferPtr<T, access, load_alignment> a, BufferPtr<T, access, other_load_alignment> b)
{
    return a.bits() < b.bits();
}

public bool operator <=<T, Access access, let load_alignment : uint, let other_load_alignment : uint>(
    BufferPtr<T, access, load_alignment> a,
    BufferPtr<T, access, other_load_alignment> b
)
{
    return a.bits() <= b.bits();
}

public bool operator><T, Access access, let load_alignment : uint, let other_load_alignment : uint>(
    BufferPtr<T, access, load_alignment> a,
    BufferPtr<T, access, other_load_alignment> b
)
{
    return a.bits() > b.bits();
}

public bool operator >=<T, Access access, let load_alignment : uint, let other_load_alignment : uint>(
    BufferPtr<T, access, load_alignment> a,
    BufferPtr<T, access, other_load_alignment> b
)
{
    return a.bits() >= b.bits();
}

[ForceInline]
public bool
operator ==<T, Access access, let load_alignment : uint>(BufferPtr<T, access, load_alignment> p, NullPtr value)
{
    return !p.is_valid();
}

[ForceInline]
public bool
operator ==<T, Access access, let load_alignment : uint>(NullPtr value, BufferPtr<T, access, load_alignment> p)
{
    return !p.is_valid();
}

[ForceInline]
public bool
operator !=<T, Access access, let load_alignment : uint>(BufferPtr<T, access, load_alignment> p, NullPtr value)
{
    return p.is_valid();
}

[ForceInline]
public bool
operator !=<T, Access access, let load_alignment : uint>(NullPtr value, BufferPtr<T, access, load_alignment> p)
{
    return p.is_valid();
}

public extension bool
{
    // Match Slang's PtrToBool cost so pointer comparisons win over bool comparisons.
    __implicit_conversion(400) public __init<T, Access access, let load_alignment : uint>(
        BufferPtr<T, access, load_alignment> p
    )
    {
        this = p.is_valid();
    }
}

public extension uint64_t
{
    /// Transport bits, not a device address on D3D12.
    public __init<T, Access access, let load_alignment : uint>(BufferPtr<T, access, load_alignment> p)
    {
        this = p.bits();
    }
}

public extension int64_t
{
    public __init<T, Access access, let load_alignment : uint>(BufferPtr<T, access, load_alignment> p)
    {
        this = int64_t(p.bits());
    }
}

// Pointer arithmetic.

/// Advance by count elements using the backend's element stride. Requires non-void T and a non-null pointer.
/// The result must remain within the same allocation or one-past-end, without overflow.
/// All arithmetic preserves load_alignment unchanged; the caller must satisfy it before access.
public BufferPtr<T, access, load_alignment>
operator +<T, Access access, let load_alignment : uint, I : __BuiltinIntegerType>(
    BufferPtr<T, access, load_alignment> p,
    I count
)
{
    return p.element_offset(count);
}

public BufferPtr<T, access, load_alignment>
operator -<T, Access access, let load_alignment : uint, I : __BuiltinIntegerType>(
    BufferPtr<T, access, load_alignment> p,
    I count
)
{
    return p.element_offset(uint64_t(0) - count.toUInt64());
}

__prefix public BufferPtr<T, access, load_alignment>
operator ++<T, Access access, let load_alignment : uint>(inout BufferPtr<T, access, load_alignment> p)
{
    p = p + 1;
    return p;
}

__postfix public BufferPtr<T, access, load_alignment>
operator ++<T, Access access, let load_alignment : uint>(inout BufferPtr<T, access, load_alignment> p)
{
    let previous = p;
    p = p + 1;
    return previous;
}

__prefix public BufferPtr<T, access, load_alignment>
operator --<T, Access access, let load_alignment : uint>(inout BufferPtr<T, access, load_alignment> p)
{
    p = p - 1;
    return p;
}

__postfix public BufferPtr<T, access, load_alignment>
operator --<T, Access access, let load_alignment : uint>(inout BufferPtr<T, access, load_alignment> p)
{
    let previous = p;
    p = p - 1;
    return previous;
}
