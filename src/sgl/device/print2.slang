// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Global debug printer buffer.
/// This is bound by the host before shader execution.
struct DebugPrinter {
    RWByteAddressBuffer buffer;
};

#ifndef SGL_ENABLE_PRINT
#define SGL_ENABLE_PRINT 1
#endif

#if SGL_ENABLE_PRINT
DebugPrinter g_debug_printer;

/// ============================================================================
/// Chunked Multi-Stream Print Buffer Format (v2)
/// ============================================================================
///
/// This module implements a lock-free, multi-stream buffer for GPU debug printing.
/// Multiple threads can write to independent streams in parallel with minimal
/// contention (atomics only on chunk allocation, not per-write).
///
/// BUFFER LAYOUT
/// -------------
/// The buffer is a single RWByteAddressBuffer with the following layout:
///
///   Offset 0:   capacity       (4 bytes) - Total buffer size in bytes
///   Offset 4:   write_pos      (4 bytes) - Current write position (atomically incremented)
///   Offset 8:   overflow_chunk (256 bytes) - Reserved chunk for overflow writes
///   Offset 264: chunks...                 - Stream chunks (256 bytes each)
///
/// The first chunk at offset 8 is reserved as an overflow zone. When the buffer
/// is full, all new allocations redirect to this chunk and are marked with
/// MAGIC_OVERFLOW. This allows write() to remain branch-free while keeping
/// all writes in-bounds. The host decoder detects and reports overflowed streams.
///
/// CHUNK FORMAT (256 bytes)
/// ------------------------
///   Offset 0:  magic        (4 bytes) - See magic values below
///   Offset 4:  next         (4 bytes) - Offset of next chunk, 0 = end of chain
///   Offset 8:  used         (4 bytes) - Bytes written to payload (0-240)
///   Offset 12: reserved     (4 bytes) - Padding for 16-byte alignment
///   Offset 16: payload    (240 bytes) - 60 x uint32 of record data
///
/// MAGIC VALUES
/// ------------
///   MAGIC_FIRST    = 0xFFFFFFFF - First chunk of a valid stream
///   MAGIC_CONTINUE = 0x00000000 - Continuation chunk of a valid stream
///   MAGIC_OVERFLOW = 0xFFFFFFFE - Chunk allocated after buffer overflow
///
/// CONSTANTS
/// ---------
///   CHUNK_SIZE     = 256
///   CHUNK_HEADER   = 16
///   CHUNK_PAYLOAD  = 240 (60 uints)
///   OVERFLOW_CHUNK = 8 (offset of reserved overflow chunk)
///
/// WRITE PROTOCOL
/// --------------
/// Creating a new stream (__init):
///   1. atomic_add(write_pos, 256) -> chunk_offset
///   2. If chunk_offset + 256 > capacity: redirect to overflow chunk
///   3. Store appropriate magic at chunk_offset+0
///   4. Store 0 at chunk_offset+4 (next = none)
///   5. Store 0 at chunk_offset+8 (used = 0)
///   6. write_offset = chunk_offset + 16
///
/// Reserving space (reserve):
///   1. If write_offset + bytes <= chunk_offset + 256: return (space available)
///   2. Store used count at current_chunk+8 (finalize)
///   3. atomic_add(write_pos, 256) -> new_chunk
///   4. If new_chunk + 256 > capacity: redirect to overflow chunk
///   5. Initialize new chunk with appropriate magic
///   6. Store new_chunk at current_chunk+4 (link)
///   7. chunk_offset = new_chunk, write_offset = new_chunk + 16
///
/// Writing data (write):
///   1. Store data at write_offset
///   2. write_offset += 4
///
/// Finishing (finish):
///   1. Store used count at chunk_offset+8
///
/// HOST DECODE PROTOCOL
/// --------------------
/// 1. Scan buffer at 256-byte intervals starting from offset 264 (skip overflow chunk)
/// 2. For each chunk with magic == MAGIC_FIRST, follow the chain
/// 3. If any chunk in chain has magic == MAGIC_OVERFLOW, discard stream and warn
/// 4. For each valid chunk: read `used` bytes from payload, follow `next`
/// 5. Decode collected payload as records
///
/// ============================================================================

namespace detail {

/// Output stream for writing debug print data.
/// Each thread should create its own OutputStream instance.
struct OutputStream {
    /// Chunk size in bytes.
    static const uint CHUNK_SIZE = 256;
    /// Chunk header size in bytes.
    static const uint CHUNK_HEADER = 16;
    /// Chunk payload size in bytes.
    static const uint CHUNK_PAYLOAD = 240;
    /// Offset of the reserved overflow chunk.
    static const uint OVERFLOW_CHUNK = 8;
    /// Magic value for first chunk of a stream.
    static const uint MAGIC_FIRST = 0xFFFFFFFFu;
    /// Magic value for continuation chunks.
    static const uint MAGIC_CONTINUE = 0x00000000u;
    /// Magic value for overflow chunks (buffer full).
    static const uint MAGIC_OVERFLOW = 0xFFFFFFFEu;

    /// Offset of current chunk in the buffer.
    uint chunk_offset;
    /// Absolute write offset in the buffer (points to next write location).
    uint write_offset;

    /// Initialize a new output stream.
    /// Allocates the first chunk and marks it as a stream start.
    __init()
    {
        // Allocate first chunk
        g_debug_printer.buffer.InterlockedAdd(4, CHUNK_SIZE, chunk_offset);

        // Check for overflow
        uint capacity = g_debug_printer.buffer.Load(0);
        uint magic = MAGIC_FIRST;
        if (chunk_offset + CHUNK_SIZE > capacity) {
            chunk_offset = OVERFLOW_CHUNK;
            magic = MAGIC_OVERFLOW;
        }

        // Initialize chunk header
        g_debug_printer.buffer.Store(chunk_offset + 0, magic);
        g_debug_printer.buffer.Store(chunk_offset + 4, 0u);           // next = none
        g_debug_printer.buffer.Store(chunk_offset + 8, 0u);           // used = 0 (set on finish)

        write_offset = chunk_offset + CHUNK_HEADER;
    }

    /// Ensure there is space for at least `bytes` more bytes in the current chunk.
    /// If not, allocates a new chunk and links it.
    /// Contract: `bytes` must be <= CHUNK_PAYLOAD (240).
    [mutating]
    void reserve(uint bytes)
    {
        if (write_offset + bytes <= chunk_offset + CHUNK_SIZE)
            return; // Enough space

        // Finalize current chunk
        g_debug_printer.buffer.Store(chunk_offset + 8, write_offset - (chunk_offset + CHUNK_HEADER));

        // Allocate new chunk
        uint new_chunk;
        g_debug_printer.buffer.InterlockedAdd(4, CHUNK_SIZE, new_chunk);

        // Check for overflow
        uint capacity = g_debug_printer.buffer.Load(0);
        uint magic = MAGIC_CONTINUE;
        if (new_chunk + CHUNK_SIZE > capacity) {
            new_chunk = OVERFLOW_CHUNK;
            magic = MAGIC_OVERFLOW;
        }

        // Initialize new chunk
        g_debug_printer.buffer.Store(new_chunk + 0, magic);
        g_debug_printer.buffer.Store(new_chunk + 4, 0u);              // next = none
        g_debug_printer.buffer.Store(new_chunk + 8, 0u);              // used = 0

        // Link current chunk to new chunk
        g_debug_printer.buffer.Store(chunk_offset + 4, new_chunk);

        // Update state
        chunk_offset = new_chunk;
        write_offset = new_chunk + CHUNK_HEADER;
    }

    /// Write a single uint to the stream.
    /// Caller must have called reserve() first to ensure space.
    [mutating]
    void write(uint data)
    {
        g_debug_printer.buffer.Store(write_offset, data);
        write_offset += 4;
    }

    /// Finalize the stream.
    /// Writes the final used count to the current chunk.
    [mutating]
    void finish()
    {
        g_debug_printer.buffer.Store(chunk_offset + 8, write_offset - (chunk_offset + CHUNK_HEADER));
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
    __init()
    {
        stream = OutputStream();
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
    void begin_format(String fmt, uint arg_count)
    {
        // Reserve space for header (1 uint) + hash (1 uint)
        stream.reserve(8);
        // Header: [kind:8][arg_count:24]
        uint header = (uint(RecordKind::format) << 24) | (arg_count & 0x00FFFFFFu);
        stream.write(header);
        stream.write(getStringHash(fmt));
    }

    /// Write a complete scalar record (header + data).
    /// \param type The scalar type.
    /// \param size_in_bytes Size of the scalar data in bytes (4 or 8).
    [mutating]
    void begin_scalar(ScalarType type, uint size_in_bytes)
    {
        // Reserve space for header (4 bytes) + data
        stream.reserve(4 + size_in_bytes);
        // Header: [kind:8][type:8][unused:16]
        uint header = (uint(RecordKind::scalar) << 24) | (uint(type) << 16);
        stream.write(header);
    }

    /// Write a vector record header and reserve space for data.
    /// Caller must then write the vector data.
    /// \param type The element type.
    /// \param count Number of elements in the vector.
    /// \param element_size_in_bytes Size of each element in bytes (4 or 8).
    [mutating]
    void begin_vector(ScalarType type, uint count, uint element_size_in_bytes)
    {
        // Reserve space for header (4 bytes) + data
        stream.reserve(4 + count * element_size_in_bytes);
        // Header: [kind:8][type:8][count:8][unused:8]
        uint header = (uint(RecordKind::vector) << 24) | (uint(type) << 16) | (count << 8);
        stream.write(header);
    }

    /// Write a matrix record header and reserve space for data.
    /// Caller must then write the matrix data.
    /// \param type The element type.
    /// \param rows Number of rows.
    /// \param cols Number of columns.
    /// \param element_size_in_bytes Size of each element in bytes (4 or 8).
    [mutating]
    void begin_matrix(ScalarType type, uint rows, uint cols, uint element_size_in_bytes)
    {
        // Reserve space for header (4 bytes) + data
        stream.reserve(4 + rows * cols * element_size_in_bytes);
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
        stream.reserve(4);
        // Header: [kind:8][unused:24]
        uint header = uint(RecordKind::begin_composite) << 24;
        stream.write(header);
    }

    /// Write an end_composite marker.
    /// This marks the end of a composite argument.
    [mutating]
    void end_composite()
    {
        stream.reserve(4);
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
    __init()
    {
        stream = detail::PrintStream();
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
    void format(String fmt)
    {
        stream.begin_format(fmt, 0);
    }

    /// Print a format string with arguments.
    [mutating]
    void format<each T : IPrintable>(String fmt, expand each T args)
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

/// Top-level print function.
/// Creates a PrintOutput and formats the message.
void print(String fmt)
{
    PrintOutput output = PrintOutput();
    output.format(fmt);
    output.finish();
}

/// Top-level print function with arguments.
/// Creates a PrintOutput and formats the message with arguments.
void print<each T : IPrintable>(String fmt, expand each T args)
{
    PrintOutput output = PrintOutput();
    output.format(fmt, expand each args);
    output.finish();
}

#endif // SGL_ENABLE_PRINT
