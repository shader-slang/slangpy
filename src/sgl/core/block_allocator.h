// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/error.h"
#include "sgl/core/macros.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace sgl {

/// Block allocator for fixed-size objects.
/// Allocates fixed-size blocks of sizeof(T) out of larger pages.
/// Thread-safe for concurrent allocations and deallocations using a mutex.
///
/// This allocator never frees pages, which means it can only
/// grow in size and never shrink.
template<typename T>
class BlockAllocator {
public:
    /// Constructor.
    /// @param blocks_per_page Number of blocks to allocate per page (default: 256).
    BlockAllocator(size_t blocks_per_page = 256)
        : m_blocks_per_page(blocks_per_page)
    {
        SGL_ASSERT(blocks_per_page > 0);
    }

    /// Destructor - frees all pages (NOT thread safe).
    ~BlockAllocator()
    {
        Page* page = m_page_list_head;
        while (page) {
            Page* next = page->next;
            std::free(page);
            page = next;
        }
    }

    SGL_NON_COPYABLE_AND_MOVABLE(BlockAllocator);

    /// Allocate a block (thread safe).
    /// @return Pointer to allocated block, or nullptr if allocation fails.
    T* allocate()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_free_list) {
            FreeBlock* block = m_free_list;
            m_free_list = block->next;
            return reinterpret_cast<T*>(block);
        }
        return allocate_from_new_page_locked();
    }

    /// Free a block (thread safe).
    /// @param ptr Pointer to block to free.
    void free(T* ptr)
    {
        if (!ptr)
            return;
        FreeBlock* block = reinterpret_cast<FreeBlock*>(ptr);
        std::lock_guard<std::mutex> lock(m_mutex);
        block->next = m_free_list;
        m_free_list = block;
    }

    /// Check if a pointer is owned by this allocator (thread safe).
    /// @param ptr Pointer to check.
    /// @return true if the pointer is within any page managed by this allocator.
    bool owns(const void* ptr) const
    {
        if (!ptr)
            return false;
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        std::lock_guard<std::mutex> lock(m_mutex);
        Page* page = m_page_list_head;
        while (page) {
            uintptr_t page_start = reinterpret_cast<uintptr_t>(page->blocks);
            uintptr_t page_end = page_start + m_blocks_per_page * sizeof(Block);
            if (addr >= page_start && addr < page_end) {
                return true;
            }
            page = page->next;
        }
        return false;
    }

    /// Reset the allocator, rebuilding the free list from all pages (NOT thread safe).
    void reset()
    {
        FreeBlock* head = nullptr;
        Page* page = m_page_list_head;
        while (page) {
            for (size_t i = 0; i < m_blocks_per_page; ++i) {
                FreeBlock* block = reinterpret_cast<FreeBlock*>(&page->blocks[i]);
                block->next = head;
                head = block;
            }
            page = page->next;
        }
        m_free_list = head;
    }

    /// Get the number of allocated pages.
    uint32_t num_pages() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_num_pages;
    }

private:
    /// Free block - stores next pointer when block is unused.
    struct FreeBlock {
        FreeBlock* next;
    };

    /// A block must be large enough to hold either T or a FreeBlock.
    union Block {
        alignas(T) uint8_t data[sizeof(T)];
        FreeBlock free_block; // Used when block is free
    };
    static_assert(sizeof(Block) >= sizeof(FreeBlock*), "Block must be large enough to hold a pointer");
    static_assert(alignof(Block) >= alignof(T), "Block alignment must be sufficient for T");

    /// A page contains multiple blocks and a link to the next page.
    /// Note: blocks[1] is a flexible array member pattern - actual size is m_blocks_per_page.
    struct Page {
        Page* next;
        Block blocks[1];
    };

    /// Allocate a new page and return a block from it.
    /// Called while m_mutex is already held.
    T* allocate_from_new_page_locked()
    {
        // Allocate a new page
        size_t page_size = sizeof(Page) + (m_blocks_per_page - 1) * sizeof(Block);
        Page* page = reinterpret_cast<Page*>(std::malloc(page_size));
        if (!page) {
            return nullptr;
        }

        // Initialize page metadata
        page->next = m_page_list_head;
        m_page_list_head = page;
        m_num_pages++;

        // Generate free list from all except first block.
        for (size_t i = 1; i < m_blocks_per_page; ++i) {
            FreeBlock* block = reinterpret_cast<FreeBlock*>(&page->blocks[i]);
            block->next = m_free_list;
            m_free_list = block;
        }

        // Return the first block
        return reinterpret_cast<T*>(&page->blocks[0]);
    }

    size_t m_blocks_per_page;
    FreeBlock* m_free_list{nullptr};
    mutable std::mutex m_mutex; // Protects all operations
    Page* m_page_list_head{nullptr};
    uint32_t m_num_pages{0};
};

/// Macro to declare block allocator support for a class.
/// The block size is automatically determined from sizeof(ClassName).
#define SGL_DECLARE_BLOCK_ALLOCATED(ClassName)                                                                         \
public:                                                                                                                \
    void* operator new(size_t count);                                                                                  \
    void operator delete(void* ptr);                                                                                   \
    /* Placement new - required because custom operator new hides inherited placement new */                           \
    void* operator new(size_t, void* p) noexcept                                                                       \
    {                                                                                                                  \
        return p;                                                                                                      \
    }                                                                                                                  \
    void operator delete(void*, void*) noexcept { }                                                                    \
                                                                                                                       \
private:                                                                                                               \
    static ::sgl::BlockAllocator<ClassName> s_allocator;

/// Macro to implement block allocator operators in .cpp file.
/// @param ClassName The class name to implement block allocation for.
/// @param BlocksPerPage Number of blocks to allocate per page.
#define SGL_IMPLEMENT_BLOCK_ALLOCATED(ClassName, BlocksPerPage)                                                        \
    ::sgl::BlockAllocator<ClassName> ClassName::s_allocator(BlocksPerPage);                                            \
                                                                                                                       \
    void* ClassName::operator new(size_t count)                                                                        \
    {                                                                                                                  \
        SGL_UNUSED(count);                                                                                             \
        return s_allocator.allocate();                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    void ClassName::operator delete(void* ptr)                                                                         \
    {                                                                                                                  \
        s_allocator.free(static_cast<ClassName*>(ptr));                                                                \
    }

} // namespace sgl
