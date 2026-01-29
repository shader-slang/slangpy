// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SGL_ENABLE_PRINT
#define SGL_ENABLE_PRINT 1
#endif

/// ============================================================================
/// Fixed-Size Message Buffer Format (v3)
/// ============================================================================
///
/// This module implements a lock-free, multi-stream buffer for GPU debug printing.
/// Each print statement allocates a fixed-size message slot. Writes within a slot
/// are completely branch-free using power-of-2 masking.
///
/// BUFFER LAYOUT
/// -------------
/// The buffer is a single RWByteAddressBuffer with the following layout:
///
///   Offset 0:      capacity      (4 bytes) - Total buffer size in bytes
///   Offset 4:      write_pos     (4 bytes) - Next slot offset (atomically incremented)
///   Offset 8:      overflow_slot (1024 bytes) - Reserved slot for overflow writes
///   Offset 1032:   slots...                - Message slots (1024 bytes each)
///
/// The first slot at offset 8 is reserved as an overflow sink. When the buffer
/// is full, new allocations redirect to this slot and set its used_size to 1.
/// This allows write() to remain completely branch-free.
///
/// SLOT FORMAT (1024 bytes, power of 2)
/// ------------------------------------
///   Offset 0:  used_size    (4 bytes) - Payload bytes used, or 0xFFFFFFFF if overflowed
///   Offset 4:  payload    (1020 bytes) - 255 x uint32 of record data
///
/// CONSTANTS
/// ---------
///   SLOT_SIZE       = 1024
///   SLOT_MASK       = 1023 (0x3FF)
///   HEADER_SIZE     = 4
///   PAYLOAD_SIZE    = 1020
///   OVERFLOW_SLOT   = 8
///   FIRST_DATA_SLOT = 1032
///   OVERFLOW_MARKER = 0xFFFFFFFF
///
/// WRITE PROTOCOL
/// --------------
/// Creating a new stream (__init):
///   1. atomic_add(write_pos, SLOT_SIZE) -> slot_offset
///   2. If slot_offset + SLOT_SIZE > capacity:
///        - Store OVERFLOW_MARKER at overflow_slot (flag overflow occurred)
///        - slot_offset = OVERFLOW_SLOT
///   3. write_offset = HEADER_SIZE (start after header)
///
/// Writing data (write) - COMPLETELY BRANCH-FREE:
///   1. buffer.Store(slot_offset + (write_offset & SLOT_MASK), data)
///   2. write_offset += 4
///
/// Finishing (finish):
///   1. If write_offset > SLOT_SIZE:
///        buffer.Store(slot_offset, OVERFLOW_MARKER)
///      Else:
///        buffer.Store(slot_offset, write_offset - HEADER_SIZE)
///
/// HOST DECODE PROTOCOL
/// --------------------
/// 1. Read global header: capacity, write_pos
/// 2. Check overflow slot (offset 8): if used_size != 0, warn about lost messages
/// 3. Scan slots at 1024-byte intervals starting from offset 1032:
///    - Read used_size at slot_offset
///    - If used_size == 0xFFFFFFFF: warn "message overflow"
///    - Else: decode used_size bytes from slot_offset + HEADER_SIZE
///
/// ============================================================================

namespace detail {

/// Output stream for writing debug print data.
/// Each thread should create its own OutputStream instance.
struct OutputStream {
    /// Slot size in bytes (power of 2).
    static const uint SLOT_SIZE = 1024;
    /// Slot mask for wrap-around writes.
    static const uint SLOT_MASK = 1023;
    /// Slot header size in bytes.
    static const uint HEADER_SIZE = 4;
    /// Slot payload size in bytes.
    static const uint PAYLOAD_SIZE = 1020;
    /// Offset of the reserved overflow slot.
    static const uint OVERFLOW_SLOT = 8;
    /// First data slot offset (after global header + overflow slot).
    static const uint FIRST_DATA_SLOT = 1032;
    /// Marker for per-message overflow (message too large for slot).
    static const uint OVERFLOW_MARKER = 0xFFFFFFFFu;

    RWByteAddressBuffer buffer;

    /// Offset of this message's slot in the buffer.
    uint slot_offset;
    /// Current write position within slot (unbounded, can exceed SLOT_SIZE).
    uint write_offset;

    /// Initialize a new output stream.
    /// Allocates a fixed-size slot for this message.
    __init(RWByteAddressBuffer buffer)
    {
        this.buffer = buffer;

        // Allocate a slot
        buffer.InterlockedAdd(4, SLOT_SIZE, slot_offset);

        // Check for buffer overflow
        uint capacity = buffer.Load(0);
        if (slot_offset + SLOT_SIZE > capacity) {
            // Flag overflow and redirect to overflow sink
            buffer.Store(OVERFLOW_SLOT, OVERFLOW_MARKER);
            slot_offset = OVERFLOW_SLOT;
        }

        // Start writing after the header
        write_offset = HEADER_SIZE;
    }

    /// Write a single uint to the stream.
    /// Completely branch-free: uses masking to wrap around within the slot.
    /// If write_offset exceeds SLOT_SIZE, data wraps and overwrites earlier data,
    /// but finish() will detect this and mark the message as overflowed.
    [mutating]
    void write(uint data)
    {
        buffer.Store(slot_offset + (write_offset & SLOT_MASK), data);
        write_offset += 4;
    }

    /// Finalize the stream.
    /// Writes the used size to the slot header, or marks as overflowed.
    [mutating]
    void finish()
    {
        if (write_offset > SLOT_SIZE) {
            // Message overflowed the slot
            buffer.Store(slot_offset, OVERFLOW_MARKER);
        } else {
            // Store actual payload size (excluding header)
            buffer.Store(slot_offset, write_offset - HEADER_SIZE);
        }
    }
};

/// ============================================================================
/// Print Record Protocol
/// ============================================================================
///
/// Records are written to an OutputStream. Each record has a 32-bit header
/// followed by type-specific data.
///
/// RECORD KINDS
/// ------------
///   format          = 0  - Format string with arguments
///   scalar          = 1  - Single value
///   vector          = 2  - Vector of values
///   matrix          = 3  - Matrix of values
///   begin_composite = 4  - Start of composite argument (wraps IPrintable output)
///   end_composite   = 5  - End of composite argument
///
/// HEADER FORMAT
/// -------------
/// FORMAT record: [kind:8][arg_count:24]
///   Followed by: [fmt_hash:32][arg0][arg1]...[argN-1]
///   Each argument is wrapped in begin_composite/end_composite markers.
///
/// SCALAR record: [kind:8][type:8][unused:16]
///   Followed by: data (1 or 2 uints depending on type)
///
/// VECTOR record: [kind:8][type:8][count:8][unused:8]
///   Followed by: data (count * element_size uints)
///
/// MATRIX record: [kind:8][type:8][rows:8][cols:8]
///   Followed by: data (rows * cols * element_size uints)
///
/// TYPE VALUES
/// -----------
///   bool_    = 0   (1 uint)
///   int8     = 1   (1 uint)
///   int16    = 2   (1 uint)
///   int32    = 3   (1 uint)
///   int64    = 4   (2 uints)
///   uint8    = 5   (1 uint)
///   uint16   = 6   (1 uint)
///   uint32   = 7   (1 uint)
///   uint64   = 8   (2 uints)
///   float16  = 9   (1 uint)
///   float32  = 10  (1 uint)
///   float64  = 11  (2 uints)
///   string   = 12  (1 uint - hash)
///
/// ============================================================================

/// Record kind values.
enum class RecordKind {
    format          = 0,
    scalar          = 1,
    vector          = 2,
    matrix          = 3,
    begin_composite = 4,
    end_composite   = 5,
};

/// Scalar type values.
enum class ScalarType {
    bool_   = 0,
    int8    = 1,
    int16   = 2,
    int32   = 3,
    int64   = 4,
    uint8   = 5,
    uint16  = 6,
    uint32  = 7,
    uint64  = 8,
    float16 = 9,
    float32 = 10,
    float64 = 11,
    string  = 12,
};

/// Print stream for writing formatted print records.
/// Wraps an OutputStream and provides methods for writing record headers.
struct PrintStream {
    /// The underlying output stream.
    OutputStream stream;

    /// Initialize a new print stream.
    __init(RWByteAddressBuffer buffer)
    {
        stream = OutputStream(buffer);
    }

    /// Finalize the stream.
    /// Must be called when done writing to ensure all data is flushed.
    [mutating]
    void finish()
    {
        stream.finish();
    }

    /// Begin a format record.
    /// Writes the format header and format string hash.
    /// Caller must then write exactly `arg_count` argument records.
    /// \param fmt The format string.
    /// \param arg_count Number of arguments that follow.
    [mutating]
    void begin_format(string fmt, uint arg_count)
    {
        // Header: [kind:8][arg_count:24]
        uint header = (uint(RecordKind::format) << 24) | (arg_count & 0x00FFFFFFu);
        stream.write(header);
        stream.write(getStringHash(fmt));
    }

    /// Write a scalar record header.
    /// Caller must then write the scalar data.
    /// \param type The scalar type.
    /// \param size_in_bytes Size of the scalar data in bytes (4 or 8).
    [mutating]
    void begin_scalar(ScalarType type, uint size_in_bytes)
    {
        // Header: [kind:8][type:8][unused:16]
        uint header = (uint(RecordKind::scalar) << 24) | (uint(type) << 16);
        stream.write(header);
    }

    /// Write a vector record header.
    /// Caller must then write the vector data.
    /// \param type The element type.
    /// \param count Number of elements in the vector.
    /// \param element_size_in_bytes Size of each element in bytes (4 or 8).
    [mutating]
    void begin_vector(ScalarType type, uint count, uint element_size_in_bytes)
    {
        // Header: [kind:8][type:8][count:8][unused:8]
        uint header = (uint(RecordKind::vector) << 24) | (uint(type) << 16) | (count << 8);
        stream.write(header);
    }

    /// Write a matrix record header.
    /// Caller must then write the matrix data.
    /// \param type The element type.
    /// \param rows Number of rows.
    /// \param cols Number of columns.
    /// \param element_size_in_bytes Size of each element in bytes (4 or 8).
    [mutating]
    void begin_matrix(ScalarType type, uint rows, uint cols, uint element_size_in_bytes)
    {
        // Header: [kind:8][type:8][rows:8][cols:8]
        uint header = (uint(RecordKind::matrix) << 24) | (uint(type) << 16) | (rows << 8) | cols;
        stream.write(header);
    }

    /// Write raw data to the stream.
    /// Used after writing a record header to write the payload.
    /// Caller must have reserved space via begin_* functions.
    [mutating]
    void write(uint data)
    {
        stream.write(data);
    }

    /// Write a begin_composite marker.
    /// This marks the start of a composite argument (wraps IPrintable output).
    [mutating]
    void begin_composite()
    {
        // Header: [kind:8][unused:24]
        uint header = uint(RecordKind::begin_composite) << 24;
        stream.write(header);
    }

    /// Write an end_composite marker.
    /// This marks the end of a composite argument.
    [mutating]
    void end_composite()
    {
        // Header: [kind:8][unused:24]
        uint header = uint(RecordKind::end_composite) << 24;
        stream.write(header);
    }
};

} // namespace detail

/// Print output for writing formatted print records with arguments.
/// This is the main interface users interact with for printing.
struct PrintOutput {
    /// The underlying print stream.
    detail::PrintStream stream;

    /// Initialize a new print output.
    __init(RWByteAddressBuffer buffer)
    {
        stream = detail::PrintStream(buffer);
    }

    /// Finalize the stream.
    /// Must be called when done writing to ensure all data is flushed.
    [mutating]
    void finish()
    {
        stream.finish();
    }

    /// Print a format string with no arguments.
    [mutating]
    void format(string fmt)
    {
        stream.begin_format(fmt, 0);
    }

    /// Print a format string with arguments.
    [mutating]
    void format<each T : IPrintable>(string fmt, expand each T args)
    {
        stream.begin_format(fmt, countof(T));
        expand print_arg(each args);
    }

    /// Helper to print a single argument, optionally wrapped in composite markers.
    [mutating]
    void print_arg<T : IPrintable>(T arg)
    {
        if (T::needs_composite_wrap()) {
            stream.begin_composite();
            arg.print(this);
            stream.end_composite();
        } else {
            arg.print(this);
        }
    }
};

/// ============================================================================
/// Printable Interfaces and Type Extensions
/// ============================================================================

/// Interface for types that can print themselves.
/// Implement this for custom types to make them printable.
interface IPrintable {
    /// Returns true if this type needs composite wrapping.
    /// Built-in types (scalars, vectors, matrices) return false since they
    /// always produce exactly one record. Custom types should return true
    /// (the safe default) unless they guarantee a single record.
    static bool needs_composite_wrap() { return true; }

    /// Print this value to the output.
    /// The implementation should write a complete record (header + data).
    void print(inout PrintOutput output);
};

namespace detail {

/// Interface for scalar types that can write their raw data.
/// This is a low-level interface used by the generic IPrintable extension.
interface IPrintableScalar {
    /// Scalar type identifier.
    static const ScalarType PRINTABLE_SCALAR_TYPE;

    /// Get the size of this scalar in bytes (4 or 8).
    static const uint PRINTABLE_SCALAR_SIZE;

    /// Write the raw scalar data (1 or 2 uints) to the stream.
    /// Caller has already reserved space and written the header.
    void write_printable_scalar(inout PrintStream stream);
};

// -----------------------------------------------------------------------------
// IPrintableScalar implementations for built-in scalar types
// -----------------------------------------------------------------------------

extension bool : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::bool_;
    static const uint PRINTABLE_SCALAR_SIZE = 4;
    void write_printable_scalar(inout PrintStream stream) { stream.write(this ? 1u : 0u); }
};

extension int8_t : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::int8;
    static const uint PRINTABLE_SCALAR_SIZE = 4;
    void write_printable_scalar(inout PrintStream stream) { stream.write(asuint(int(this))); }
};

extension int16_t : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::int16;
    static const uint PRINTABLE_SCALAR_SIZE = 4;
    void write_printable_scalar(inout PrintStream stream) { stream.write(asuint(int(this))); }
};

extension int32_t : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::int32;
    static const uint PRINTABLE_SCALAR_SIZE = 4;
    void write_printable_scalar(inout PrintStream stream) { stream.write(asuint(this)); }
};

extension int64_t : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::int64;
    static const uint PRINTABLE_SCALAR_SIZE = 8;
    void write_printable_scalar(inout PrintStream stream) {
        stream.write(uint(this & 0xFFFFFFFF));
        stream.write(uint(this >> 32));
    }
};

extension uint8_t : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::uint8;
    static const uint PRINTABLE_SCALAR_SIZE = 4;
    void write_printable_scalar(inout PrintStream stream) { stream.write(uint(this)); }
};

extension uint16_t : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::uint16;
    static const uint PRINTABLE_SCALAR_SIZE = 4;
    void write_printable_scalar(inout PrintStream stream) { stream.write(uint(this)); }
};

extension uint32_t : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::uint32;
    static const uint PRINTABLE_SCALAR_SIZE = 4;
    void write_printable_scalar(inout PrintStream stream) { stream.write(this); }
};

extension uint64_t : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::uint64;
    static const uint PRINTABLE_SCALAR_SIZE = 8;
    void write_printable_scalar(inout PrintStream stream) {
        stream.write(uint(this & 0xFFFFFFFFu));
        stream.write(uint(this >> 32));
    }
};

extension float16_t : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::float16;
    static const uint PRINTABLE_SCALAR_SIZE = 4;
    void write_printable_scalar(inout PrintStream stream) { stream.write(uint(asuint16(this))); }
};

extension float : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::float32;
    static const uint PRINTABLE_SCALAR_SIZE = 4;
    void write_printable_scalar(inout PrintStream stream) { stream.write(asuint(this)); }
};

extension double : IPrintableScalar {
    static const ScalarType PRINTABLE_SCALAR_TYPE = ScalarType::float64;
    static const uint PRINTABLE_SCALAR_SIZE = 8;
    void write_printable_scalar(inout PrintStream stream) {
        uint lo, hi;
        asuint(this, lo, hi);
        stream.write(lo);
        stream.write(hi);
    }
};

} // namespace detail

// -----------------------------------------------------------------------------
// Generic IPrintable extension for all IPrintableScalar types
// -----------------------------------------------------------------------------

extension<T : detail::IPrintableScalar> T : IPrintable {
    override static bool needs_composite_wrap() { return false; }
    void print(inout PrintOutput output) {
        output.stream.begin_scalar(T::PRINTABLE_SCALAR_TYPE, T::PRINTABLE_SCALAR_SIZE);
        this.write_printable_scalar(output.stream);
    }
};

// -----------------------------------------------------------------------------
// IPrintable implementations for vector types
// -----------------------------------------------------------------------------

extension<T : detail::IPrintableScalar, let N : int> vector<T, N> : IPrintable {
    override static bool needs_composite_wrap() { return false; }
    void print(inout PrintOutput output) {
        output.stream.begin_vector(T::PRINTABLE_SCALAR_TYPE, N, T::PRINTABLE_SCALAR_SIZE);
        for (int i = 0; i < N; ++i) {
            this[i].write_printable_scalar(output.stream);
        }
    }
};

// -----------------------------------------------------------------------------
// IPrintable implementations for matrix types
// -----------------------------------------------------------------------------

extension<T : detail::IPrintableScalar, let R : int, let C : int> matrix<T, R, C> : IPrintable {
    override static bool needs_composite_wrap() { return false; }
    void print(inout PrintOutput output) {
        output.stream.begin_matrix(T::PRINTABLE_SCALAR_TYPE, R, C, T::PRINTABLE_SCALAR_SIZE);
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                this[r][c].write_printable_scalar(output.stream);
            }
        }
    }
};

// -----------------------------------------------------------------------------
// IPrintable implementation for string type
// -----------------------------------------------------------------------------

extension string : IPrintable {
    override static bool needs_composite_wrap() { return false; }
    void print(inout PrintOutput output) {
        output.stream.begin_scalar(detail::ScalarType::string, 4);
        output.stream.write(getStringHash(this));
    }
};

// Slang's native string has very limited support, notably,
// cannot be realibly passed out of functions. The work-around is to use
// result of `getStringHash` instead of the actual string literal and pass that instead.
// PrintableString wraps the hash so it is strongly typed and can be extended.
struct PrintableString {
    uint hash;

    __init() { hash = getStringHash(""); }
    __implicit_conversion(1) __init(string str) { hash = getStringHash(str); }
}

extension PrintableString : IPrintable {
    override static bool needs_composite_wrap() { return false; }
    void print(inout PrintOutput output) {
        output.stream.begin_scalar(detail::ScalarType::string, 4);
        output.stream.write(hash);
    }
};

#if SGL_ENABLE_PRINT

/// Global debug printer buffer.
/// This is bound by the host before shader execution.
struct DebugPrinter {
    RWByteAddressBuffer buffer;
};

DebugPrinter g_debug_printer;

/// Top-level print function.
void print(string fmt)
{
    PrintOutput output = PrintOutput(g_debug_printer.buffer);
    output.format(fmt);
    output.finish();
}

/// Top-level print function with arguments.
void print<each T : IPrintable>(string fmt, expand each T args)
{
    PrintOutput output = PrintOutput(g_debug_printer.buffer);
    output.format(fmt, expand each args);
    output.finish();
}

#else

void print(string fmt) { }
void print<each T : IPrintable>(string fmt, expand each T args) { }

#endif // SGL_ENABLE_PRINT
