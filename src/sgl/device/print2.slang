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
/// Multi-Stream Print Buffer Format (v2)
/// ============================================================================
///
/// This module implements a lock-free, multi-stream buffer for GPU debug printing.
/// Multiple threads can write to independent streams in parallel with minimal
/// contention (only a single atomic add per entry).
///
/// BUFFER LAYOUT
/// -------------
/// The buffer is a single RWByteAddressBuffer with the following layout:
///
///   Offset 0:  [capacity:32]     - Total buffer size in bytes
///   Offset 4:  [write_pos:32]    - Current write position (atomically incremented)
///   Offset 8+: [entries...]      - Stream entries
///
/// ENTRY FORMAT
/// ------------
/// Each entry is 8 bytes (2 x uint32):
///
///   [header:32][payload:32]
///
/// The header field serves as a link to the next entry in the stream:
///   - header = 0          : End of stream (no more entries)
///   - header = offset     : Offset to next entry in this stream
///
/// STREAM FORMAT
/// -------------
/// A stream begins with a 16-byte header (4 x uint32):
///
///   [0xFFFFFFFF][0xFFFFFFFF][link:32][unused:32]
///    └─── sentinel pair ───┘ └── dummy entry ──┘
///
/// The sentinel pair (two consecutive 0xFFFFFFFF values) identifies the start
/// of a new stream. The dummy entry provides a header slot for linking to the
/// first real data entry.
///
/// Subsequent entries are 8 bytes each:
///
///   [link:32][data:32]
///
/// WRITE PROTOCOL
/// --------------
/// Creating a new stream:
///   1. atomic_add(write_pos, 16) -> offset
///   2. Store 0xFFFFFFFF at offset+0  (sentinel)
///   3. Store 0xFFFFFFFF at offset+4  (sentinel)
///   4. Store 0 at offset+8           (dummy header - end of chain)
///   5. Store 0 at offset+12          (dummy payload - unused)
///   6. Set prev = offset+8           (points to dummy entry's header)
///
/// Writing a data uint:
///   1. atomic_add(write_pos, 8) -> offset
///   2. Store offset at buffer[prev]  (link previous entry to this one)
///   3. Store 0 at offset+0           (header - end of chain for now)
///   4. Store data at offset+4        (payload)
///   5. Set prev = offset             (update for next write)
///
/// This gives us a uniform write path: 1 atomic add, 3 stores, no branches.
///
/// HOST DECODE PROTOCOL
/// --------------------
/// 1. Linear scan through buffer looking for sentinel pairs (0xFFFFFFFF, 0xFFFFFFFF)
/// 2. For each sentinel found:
///    a. Read the link from the dummy entry (offset+8)
///    b. Follow the chain, collecting payload values until link=0
/// 3. Process collected stream data
///
/// OVERFLOW HANDLING
/// -----------------
/// If write_pos exceeds capacity, writes will go out of bounds. The host
/// should check if write_pos > capacity and report overflow to the user.
///
/// ============================================================================

namespace debug_print {

/// Output stream for writing debug print data.
/// Each thread should create its own OutputStream instance.
struct OutputStream {
    /// Offset to the previous entry's header (where we'll write the next link).
    uint prev;

    /// Initialize a new output stream.
    /// This allocates the stream header (sentinel + dummy entry) in the buffer.
    __init()
    {
        // Allocate 16 bytes for stream header
        uint offset;
        g_debug_printer.buffer.InterlockedAdd(4, 16, offset);

        // Write sentinel pair
        g_debug_printer.buffer.Store(offset + 0, 0xFFFFFFFFu);
        g_debug_printer.buffer.Store(offset + 4, 0xFFFFFFFFu);

        // Write dummy entry (end of chain marker)
        g_debug_printer.buffer.Store(offset + 8, 0u);
        g_debug_printer.buffer.Store(offset + 12, 0u);

        // Point prev to dummy entry's header slot
        prev = offset + 8;
    }

    /// Write a single uint to the stream.
    /// This is the core write operation - all other writes decompose to this.
    [mutating]
    void write(uint data)
    {
        // Allocate 8 bytes for new entry
        uint offset;
        g_debug_printer.buffer.InterlockedAdd(4, 8, offset);

        // Link previous entry to this one
        g_debug_printer.buffer.Store(prev, offset);

        // Write new entry (end of chain for now)
        g_debug_printer.buffer.Store(offset, 0u);
        g_debug_printer.buffer.Store(offset + 4, data);

        // Update prev for next write
        prev = offset;
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
///   format = 0  - Format string with arguments
///   scalar = 1  - Single value
///   vector = 2  - Vector of values
///   matrix = 3  - Matrix of values
///
/// HEADER FORMAT
/// -------------
/// FORMAT record: [kind:8][arg_count:24]
///   Followed by: [fmt_hash:32][arg0][arg1]...[argN-1]
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
    format = 0,
    scalar = 1,
    vector = 2,
    matrix = 3,
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

    /// Begin a format record.
    /// Writes the format header and format string hash.
    /// Caller must then write exactly `arg_count` argument records.
    /// \param fmt The format string.
    /// \param arg_count Number of arguments that follow.
    [mutating]
    void begin_format(String fmt, uint arg_count)
    {
        // Header: [kind:8][arg_count:24]
        uint header = (uint(RecordKind::format) << 24) | (arg_count & 0x00FFFFFFu);
        stream.write(header);
        stream.write(getStringHash(fmt));
    }

    /// Write a scalar record header.
    /// Caller must then write the scalar data (1 or 2 uints depending on type).
    /// \param type The scalar type (from ScalarType enum).
    [mutating]
    void write_scalar_header(ScalarType type)
    {
        // Header: [kind:8][type:8][unused:16]
        uint header = (uint(RecordKind::scalar) << 24) | (uint(type) << 16);
        stream.write(header);
    }

    /// Write a vector record header.
    /// Caller must then write the vector data (count * element_size uints).
    /// \param type The element type (from ScalarType enum).
    /// \param count Number of elements in the vector.
    [mutating]
    void write_vector_header(ScalarType type, uint count)
    {
        // Header: [kind:8][type:8][count:8][unused:8]
        uint header = (uint(RecordKind::vector) << 24) | (uint(type) << 16) | (count << 8);
        stream.write(header);
    }

    /// Write a matrix record header.
    /// Caller must then write the matrix data (rows * cols * element_size uints).
    /// \param type The element type (from ScalarType enum).
    /// \param rows Number of rows.
    /// \param cols Number of columns.
    [mutating]
    void write_matrix_header(ScalarType type, uint rows, uint cols)
    {
        // Header: [kind:8][type:8][rows:8][cols:8]
        uint header = (uint(RecordKind::matrix) << 24) | (uint(type) << 16) | (rows << 8) | cols;
        stream.write(header);
    }

    /// Write raw data to the stream.
    /// Used after writing a record header to write the payload.
    [mutating]
    void write(uint data)
    {
        stream.write(data);
    }
};

} // namespace debug_print

#endif // SGL_ENABLE_PRINT
