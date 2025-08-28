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

void process_arg<T : IPrintable>(T arg, inout uint arg_count, inout uint total_data_count)
{
    arg_count++;
    total_data_count += arg.printable_data_count();
}

[ForceInline]
void print<each T : IPrintable>(IPrintOutput output, String fmt, expand each T args)
{
    uint arg_count = 0;
    uint total_data_count = 0;
    expand process_arg(each args, arg_count, total_data_count);
    uint offset;
    if (!output.write_msg(fmt, arg_count, total_data_count, offset))
        return;
    expand output.write_arg(offset, each args);
}

struct DebugPrinter : IPrintOutput {
    /// Single buffer that contains the print messages.
    /// First 4 bytes store the capacity of the buffer (including the header).
    /// Next 4 bytes store the current size of the buffer (excluding the header).
    /// The rest of the buffer is used to store messages.
    RWByteAddressBuffer buffer;

    [ForceInline]
    bool write_msg(String fmt, uint arg_count, uint total_data_count, out uint offset)
    {
        // Compute the size of the message.
        uint size = (3 + arg_count + total_data_count) * sizeof(uint);

        // Get the capacity of the buffer.
        uint capacity = buffer.Load(0);

        // Reserve space for the message.
        buffer.InterlockedAdd(4, size, offset);

        // Skip the header.
        offset += 8;

        if (offset + size > capacity - 4) {
            // Write sentinel value indicating that we have a buffer overlow.
            buffer.Store(offset, 0xffffffff);
            return false;
        }

        // Store the message header.
        buffer.Store(offset, size);
#ifdef __TARGET_METAL__
        // Workaround for Slang issue https://github.com/shader-slang/slang/issues/7534
        buffer.Store(offset + 4, uint(getStringHash(fmt)));
#else
        buffer.Store(offset + 4, getStringHash(fmt));
#endif
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

detail::DebugPrinter g_debug_printer;

[ForceInline]
void print<each T : detail::IPrintable>(String fmt, expand each T args)
{
    detail::print(g_debug_printer, fmt, args);
}

#else // SGL_ENABLE_PRINT

[ForceInline]
void print<each T : detail::IPrintable>(String fmt, expand each T args) {}

#endif // SGL_ENABLE_PRINT
