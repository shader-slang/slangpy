// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
/// This gives us a uniform write path: 3 stores, no branches.
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
#endif
