// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"

#include <atomic>
#include <filesystem>
#include <span>
#include <vector>
#include <optional>

// Forward declaration
struct MDB_env;

namespace sgl {

/// Exception class for LMDB errors.
class SGL_API LMDBException : public std::runtime_error {
public:
    LMDBException(const std::string& what, int error)
        : std::runtime_error(what)
        , m_error(error)
    {
    }

    int error() const { return m_error; }

private:
    int m_error;
};


/// \brief LMDB-based persistent cache.
/// This class provides a simple key-value cache that stores its data in an LMDB database on disk.
/// It supports basic operations such as setting, getting, and deleting entries.
/// Eviction uses an LRU policy and is triggered when the cache size exceeds the eviction threshold.
class SGL_API LMDBCache : public Object {
    SGL_OBJECT(LMDBCache)
public:
    using WriteValueFunc = void (*)(const void* data, size_t size, void* user_data);

    struct Options {
        /// Maximum size of the cache on disk.
        size_t max_size{64ull * 1024 * 1024};
        /// Eviction threshold in percent (0-100). When the cache size exceeds this
        /// percentage of the maximum size, eviction is triggered.
        uint32_t eviction_threshold = 80;
        /// Eviction target in percent (0-100). When eviction is triggered, entries
        /// are evicted until the cache size is below this percentage of the maximum size.
        uint32_t eviction_target = 60;
        /// Disable synchronous writes to improve performance at the cost of potential data loss
        /// in case of a crash. This is equivalent to opening the LMDB environment with the
        /// `MDB_NOSYNC` flag.
        bool nosync = true;
    };

    struct Usage {
        /// Reserved size in bytes (maximum size on disk).
        size_t reserved_size{0};
        /// Committed size in bytes (current size on disk).
        size_t committed_size{0};
        /// Used size in bytes (storing active entries).
        size_t used_size{0};
    };

    struct Stats {
        /// Number of entries in the cache.
        uint64_t entries{0};
        /// Total size of all entries in the cache.
        uint64_t size{0};
        /// Eviction count (number of entries evicted since opening).
        uint64_t evictions{0};
    };

    /// Constructor.
    /// Open the cache at the specified path.
    /// Throws on error.
    /// \param path Path to the cache directory.
    /// \param options Cache options.
    LMDBCache(const std::filesystem::path& path, std::optional<Options> options = {});

    /// Destructor.
    ~LMDBCache() override;

    /// Set a value in the cache.
    /// Throws on error.
    /// \param key_data Pointer to the key data.
    /// \param key_size Size of the key data.
    /// \param value_data Pointer to the value data.
    /// \param value_size Size of the value data.
    void set(const void* key_data, size_t key_size, const void* value_data, size_t value_size);

    /// Get a value from the cache.
    /// Throws on error.
    /// \param key_data Pointer to the key data.
    /// \param key_size Size of the key data.
    /// \param write_value_func Function to write the value data.
    /// \param user_data User data passed to the write_value_func.
    /// \return True if the key was found, false otherwise.
    bool get(const void* key_data, size_t key_size, WriteValueFunc write_value_func, void* user_data = nullptr);

    /// Delete a value from the cache.
    /// Throws on error.
    /// \param key_data Pointer to the key data.
    /// \param key_size Size of the key data.
    /// \return True if the key was found and deleted, false if the key was not found.
    bool del(const void* key_data, size_t key_size);

    /// Set a value in the cache.
    /// Throws on error.
    /// \param key Key.
    /// \param value Value.
    inline void set(std::span<const uint8_t> key, std::span<const uint8_t> value)
    {
        set(key.data(), key.size(), value.data(), value.size());
    }

    /// Get a value from the cache.
    /// Throws on error.
    /// \param key Key.
    /// \param value Vector to store the value.
    /// \return True if the key was found, false otherwise.
    inline bool get(std::span<const uint8_t> key, std::vector<uint8_t>& value)
    {
        return get(
            key.data(),
            key.size(),
            [](const void* data, size_t size, void* user_data)
            {
                reinterpret_cast<std::vector<uint8_t>*>(user_data)->assign(
                    static_cast<const uint8_t*>(data),
                    static_cast<const uint8_t*>(data) + size
                );
            },
            &value
        );
    }

    /// Delete a value from the cache.
    /// Throws on error.
    /// \param key Key.
    /// \return True if the key was found and deleted, false if the key was not found.
    inline bool del(std::span<const uint8_t> key) { return del(key.data(), key.size()); }

    Usage usage() const;
    Stats stats() const;

private:
    void evict();

    struct DB {
        MDB_env* env{nullptr};
        unsigned int dbi_data{0};
        unsigned int dbi_meta{0};
    };

    static DB open_db(const std::filesystem::path& path, const Options& options);
    static void close_db(DB db);

    DB m_db;

    size_t m_max_key_size{0};

    std::atomic<uint64_t> m_evictions{0};

    size_t m_eviction_threshold_size{0};
    size_t m_eviction_target_size{0};

    SGL_NON_COPYABLE_AND_MOVABLE(LMDBCache);

    friend struct DBCacheItem;
};

} // namespace sgl
