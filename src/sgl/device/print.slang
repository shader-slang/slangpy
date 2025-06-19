// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SGL_ENABLE_PRINT
#define SGL_ENABLE_PRINT 0
#endif

namespace detail {

enum class Kind {
    scalar,
    vector,
    matrix,
};

/// Printable primitive types.
enum class Type {
    boolean,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
};

struct Layout {
    Kind kind;
    Type type;
    uint rows;
    uint cols;
};

/// Interface to print output.
interface IPrintOutput {
    /// Write a message and reserve space for writing the arguments.
    /// \param fmt The format string.
    /// \param arg_count The number of arguments.
    /// \param total_data_count The total number of data elements in all arguments.
    /// \param[out] offset Offset into the output buffer. This needs to be passed to \c write_arg calls.
    /// \return True if the message was written successfully, false if there was not enough space.
    /// \TODO: Note that the implementation of this interface is using RWByteAddressBuffer::GetDimensions which
    /// has the requirement that the Metal is not supported. Slang requires the capability requirement on the interface,
    /// so we will add this requirement here to prevent Slang generating invalid code for Metal.
    /// See: https://github.com/shader-slang/slangpy/issues/302
    [require(cpp_cuda_glsl_hlsl_spirv_wgsl, structuredbuffer_rw)]
    bool write_msg(String fmt, uint arg_count, uint total_data_count, out uint offset);

    /// Write a single printable argument.
    /// \param[in,out] offset Offset into the output buffer.
    void write_arg(inout uint offset, IPrintable arg);
};

/// Interface for printable types.
interface IPrintable {
    [ForceInline]
    static Layout printable_layout();

    /// The total number of data elements (each element is a \c uint).
    [ForceInline]
    static uint printable_data_count();

    /// Get a printable data element.
    /// \param index The index of the data element in range [0..printable_data_count()).
    [ForceInline]
    uint get_printable_data(uint index);
};

// clang-format off
extension bool : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::boolean, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 1; }
    [ForceInline]
    uint get_printable_data(uint index) { return asuint(this); }
};

extension int8_t : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::int8, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 1; }
    [ForceInline]
    uint get_printable_data(uint index) { return asuint(this); }
};

extension int16_t : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::int16, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 1; }
    [ForceInline]
    uint get_printable_data(uint index) { return asuint(this); }
};

extension int32_t : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::int32, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 1; }
    [ForceInline]
    uint get_printable_data(uint index) { return asuint(this); }
};

extension int64_t : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::int64, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 2; }
    [ForceInline]
    uint get_printable_data(uint index) { return index == 0 ? uint(this & 0xffffffff) : uint(this >> 32); }
};

extension uint8_t : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::uint8, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 1; }
    [ForceInline]
    uint get_printable_data(uint index) { return this; }
};

extension uint16_t : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::uint16, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 1; }
    [ForceInline]
    uint get_printable_data(uint index) { return this; }
};

extension uint32_t : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::uint32, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 1; }
    [ForceInline]
    uint get_printable_data(uint index) { return this; }
};

extension uint64_t : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::uint64, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 2; }
    [ForceInline]
    uint get_printable_data(uint index) { return index == 0 ? uint(this & 0xffffffffu) : uint(this >> 32); }
};

extension float16_t : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::float16, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 1; }
    [ForceInline]
    uint get_printable_data(uint index) { return asuint16(this); }
}

extension float : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::float32, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 1; }
    [ForceInline]
    uint get_printable_data(uint index) { return asuint(this); }
}

extension double : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::scalar, Type::float64, 0, 0); }
    [ForceInline]
    static uint printable_data_count() { return 2; }
    [ForceInline]
    uint get_printable_data(uint index) { uint2 data; asuint(this, data.x, data.y); return data[index]; }
}

__generic<T : IPrintable, let N : int>
extension vector<T, N> : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::vector, T::printable_layout().type, N, 0); }
    [ForceInline]
    static uint printable_data_count() { return N * T::printable_data_count(); }
    [ForceInline]
    uint get_printable_data(uint index) { return this[index / T::printable_data_count()].get_printable_data(index % T::printable_data_count()); }
};

__generic<T : IPrintable, let R : int, let C : int, let L : int>
extension matrix<T, R, C, L> : IPrintable {
    [ForceInline]
    static Layout printable_layout() { return Layout(Kind::matrix, T::printable_layout().type, R, C); }
    static uint printable_data_count() { return R * C * T::printable_data_count(); }
    [ForceInline]
    uint get_printable_data(uint index) {
        uint element_index = index / T::printable_data_count();
        uint row = element_index / C;
        uint col = element_index % C;
        return this[row][col].get_printable_data(index % T::printable_data_count());
    }
};
// clang-format on

[ForceInline]
void print(IPrintOutput output, String fmt)
{
    uint offset;
    output.write_msg(fmt, 0, 0, offset);
}

[ForceInline]
void print(IPrintOutput output, String fmt, IPrintable arg0)
{
    uint total_data_count = arg0.printable_data_count();
    uint offset;
    if (!output.write_msg(fmt, 1, total_data_count, offset))
        return;
    output.write_arg(offset, arg0);
}

[ForceInline]
void print(IPrintOutput output, String fmt, IPrintable arg0, IPrintable arg1)
{
    uint total_data_count = arg0.printable_data_count() + arg1.printable_data_count();
    uint offset;
    if (!output.write_msg(fmt, 2, total_data_count, offset))
        return;
    output.write_arg(offset, arg0);
    output.write_arg(offset, arg1);
}

[ForceInline]
void print(IPrintOutput output, String fmt, IPrintable arg0, IPrintable arg1, IPrintable arg2)
{
    uint total_data_count = arg0.printable_data_count() + arg1.printable_data_count() + arg2.printable_data_count();
    uint offset;
    if (!output.write_msg(fmt, 3, total_data_count, offset))
        return;
    output.write_arg(offset, arg0);
    output.write_arg(offset, arg1);
    output.write_arg(offset, arg2);
}

[ForceInline]
void print(IPrintOutput output, String fmt, IPrintable arg0, IPrintable arg1, IPrintable arg2, IPrintable arg3)
{
    uint total_data_count = arg0.printable_data_count() + arg1.printable_data_count() + arg2.printable_data_count()
        + arg3.printable_data_count();
    uint offset;
    if (!output.write_msg(fmt, 4, total_data_count, offset))
        return;
    output.write_arg(offset, arg0);
    output.write_arg(offset, arg1);
    output.write_arg(offset, arg2);
    output.write_arg(offset, arg3);
}

[ForceInline]
void print(
    IPrintOutput output,
    String fmt,
    IPrintable arg0,
    IPrintable arg1,
    IPrintable arg2,
    IPrintable arg3,
    IPrintable arg4
)
{
    uint total_data_count = arg0.printable_data_count() + arg1.printable_data_count() + arg2.printable_data_count()
        + arg3.printable_data_count() + arg4.printable_data_count();
    ;
    uint offset;
    if (!output.write_msg(fmt, 5, total_data_count, offset))
        return;
    output.write_arg(offset, arg0);
    output.write_arg(offset, arg1);
    output.write_arg(offset, arg2);
    output.write_arg(offset, arg3);
    output.write_arg(offset, arg4);
}

[ForceInline]
void print(
    IPrintOutput output,
    String fmt,
    IPrintable arg0,
    IPrintable arg1,
    IPrintable arg2,
    IPrintable arg3,
    IPrintable arg4,
    IPrintable arg5
)
{
    uint total_data_count = arg0.printable_data_count() + arg1.printable_data_count() + arg2.printable_data_count()
        + arg3.printable_data_count() + arg4.printable_data_count() + arg5.printable_data_count();
    ;
    uint offset;
    if (!output.write_msg(fmt, 6, total_data_count, offset))
        return;
    output.write_arg(offset, arg0);
    output.write_arg(offset, arg1);
    output.write_arg(offset, arg2);
    output.write_arg(offset, arg3);
    output.write_arg(offset, arg4);
    output.write_arg(offset, arg5);
}

[ForceInline]
void print(
    IPrintOutput output,
    String fmt,
    IPrintable arg0,
    IPrintable arg1,
    IPrintable arg2,
    IPrintable arg3,
    IPrintable arg4,
    IPrintable arg5,
    IPrintable arg6
)
{
    uint total_data_count = arg0.printable_data_count() + arg1.printable_data_count() + arg2.printable_data_count()
        + arg3.printable_data_count() + arg4.printable_data_count() + arg5.printable_data_count()
        + arg6.printable_data_count();
    ;
    uint offset;
    if (!output.write_msg(fmt, 7, total_data_count, offset))
        return;
    output.write_arg(offset, arg0);
    output.write_arg(offset, arg1);
    output.write_arg(offset, arg2);
    output.write_arg(offset, arg3);
    output.write_arg(offset, arg4);
    output.write_arg(offset, arg5);
    output.write_arg(offset, arg6);
}

[ForceInline]
void print(
    IPrintOutput output,
    String fmt,
    IPrintable arg0,
    IPrintable arg1,
    IPrintable arg2,
    IPrintable arg3,
    IPrintable arg4,
    IPrintable arg5,
    IPrintable arg6,
    IPrintable arg7
)
{
    uint total_data_count = arg0.printable_data_count() + arg1.printable_data_count() + arg2.printable_data_count()
        + arg3.printable_data_count() + arg4.printable_data_count() + arg5.printable_data_count()
        + arg6.printable_data_count() + arg7.printable_data_count();
    ;
    uint offset;
    if (!output.write_msg(fmt, 8, total_data_count, offset))
        return;
    output.write_arg(offset, arg0);
    output.write_arg(offset, arg1);
    output.write_arg(offset, arg2);
    output.write_arg(offset, arg3);
    output.write_arg(offset, arg4);
    output.write_arg(offset, arg5);
    output.write_arg(offset, arg6);
    output.write_arg(offset, arg7);
}

struct DebugPrinter : IPrintOutput {

    /// Single buffer that contains the print messages.
    /// The first 4 bytes of the buffer store the size of the data following.
    RWByteAddressBuffer buffer;

    [ForceInline]
    bool write_msg(String fmt, uint arg_count, uint total_data_count, out uint offset)
    {
        uint buffer_capacity;
        buffer.GetDimensions(buffer_capacity);

        // Compute the size of the message.
        uint size = (3 + arg_count + total_data_count) * sizeof(uint);

        // Reserve space for the message.
        offset = 0;
        buffer.InterlockedAdd(0, size, offset);
        offset += 4;
        if (offset + size > buffer_capacity - 4) {
            // Write sentinel value indicating that we have a buffer overlow.
            buffer.Store(offset, 0xffffffff);
            return false;
        }

        buffer.Store(offset, size);
        buffer.Store(offset + 4, getStringHash(fmt));
        buffer.Store(offset + 8, arg_count);
        offset += 12;
        return true;
    }

    [ForceInline]
    void write_arg(inout uint offset, IPrintable arg)
    {
        Layout layout = arg.printable_layout();
        uint size = (1 + arg.printable_data_count()) * sizeof(uint);
        // Header: [kind:4][type:4][rows:4][cols:4][size:16]
        uint header
            = (uint(layout.kind) << 28) | (uint(layout.type) << 24) | (layout.rows << 20) | (layout.cols << 16) | size;
        buffer.Store(offset, header);
        for (uint i = 0; i < arg.printable_data_count(); ++i)
            buffer.Store(offset + (i + 1) * 4, arg.get_printable_data(i));
        offset += size;
    }
};

} // namespace detail

#if SGL_ENABLE_PRINT

ParameterBlock<detail::DebugPrinter> g_debug_printer;

// clang-format off
[ForceInline] void print(String fmt) { detail::print(g_debug_printer, fmt); }
[ForceInline] void print(String fmt, detail::IPrintable arg0) { detail::print(g_debug_printer, fmt, arg0); }
[ForceInline] void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1) { detail::print(g_debug_printer, fmt, arg0, arg1); }
[ForceInline] void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2) { detail::print(g_debug_printer, fmt, arg0, arg1, arg2); }
[ForceInline] void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3) { detail::print(g_debug_printer, fmt, arg0, arg1, arg2, arg3); }
[ForceInline] void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3, detail::IPrintable arg4) { detail::print(g_debug_printer, fmt, arg0, arg1, arg2, arg3, arg4); }
[ForceInline] void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3, detail::IPrintable arg4, detail::IPrintable arg5) { detail::print(g_debug_printer, fmt, arg0, arg1, arg2, arg3, arg4, arg5); }
[ForceInline] void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3, detail::IPrintable arg4, detail::IPrintable arg5, detail::IPrintable arg6) { detail::print(g_debug_printer, fmt, arg0, arg1, arg2, arg3, arg4, arg5, arg6); }
[ForceInline] void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3, detail::IPrintable arg4, detail::IPrintable arg5, detail::IPrintable arg6, detail::IPrintable arg7) { detail::print(g_debug_printer, fmt, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7); }
// clang-format on

#else // SGL_ENABLE_PRINT

// clang-format off
void print(String fmt) { }
void print(String fmt, detail::IPrintable arg0) { }
void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1) { }
void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2) { }
void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3) { }
void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3, detail::IPrintable arg4) { }
void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3, detail::IPrintable arg4, detail::IPrintable arg5) { }
void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3, detail::IPrintable arg4, detail::IPrintable arg5, detail::IPrintable arg6) { }
void print(String fmt, detail::IPrintable arg0, detail::IPrintable arg1, detail::IPrintable arg2, detail::IPrintable arg3, detail::IPrintable arg4, detail::IPrintable arg5, detail::IPrintable arg6, detail::IPrintable arg7) { }
// clang-format on

#endif // SGL_ENABLE_PRINT
