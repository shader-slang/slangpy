// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"

#include <atomic>
#include <filesystem>
#include <span>

// Forward declaration
struct MDB_env;

namespace sgl {

/// Persistent cache implementation using LMDB as a backend.
/// This cache supports storing key-value pairs on disk.
class SGL_API PersistentCache : public Object {
    SGL_OBJECT(PersistentCache)
public:
    using ValueCallback = void (*)(const void* data, size_t size, void* user_data);

    struct Options {
        /// Maximum size of the cache on disk.
        size_t max_disk_size{64ull * 1024 * 1024};

        /// Automatically evict entries when the cache is full.
        bool auto_evict{true};
        size_t auto_evict_target_size{32ull * 1024 * 1024};
        size_t auto_evict_after_bytes{1024 * 1024};

        Options() { }
    };

    struct Stats {
        /// Number of entries in the cache.
        uint64_t entries{0};
        /// Total size of all entries in the cache.
        uint64_t size{0};
    };

    PersistentCache();
    ~PersistentCache() override;

    const std::string& last_error() const { return m_last_error; }

    /// Open the cache at the specified path.
    /// \param path Path to the cache directory (created if it does not exist).
    /// \param options Cache options.
    /// \return True if the cache was opened successfully, false otherwise.
    bool open(const std::filesystem::path& path, const Options& options = Options());

    /// Close the cache.
    void close();

    /// Set a value in the cache.
    /// \param key_data Key data.
    /// \param key_size Key size.
    /// \param value_data Value data.
    /// \param value_size Value size.
    /// \return True if the value was set successfully, false otherwise.
    bool set(const void* key_data, size_t key_size, const void* value_data, size_t value_size);

    /// Get a value from the cache.
    /// \param key_data Key data.
    /// \param key_size Key size.
    /// \param value_callback Callback called with the value data if found.
    /// \param user_data User data to pass to the value_callback
    /// \return True if the value was found and written, false otherwise.
    bool get(const void* key_data, size_t key_size, ValueCallback value_callback, void* user_data = nullptr);

    /// Delete a value from the cache.
    /// \param key_data Key data.
    /// \param key_size Key size.
    /// \return True if the value was deleted successfully, false otherwise.
    bool del(const void* key_data, size_t key_size);

    /// Set a value in the cache using std::span.
    /// \param key Key as a span of bytes.
    /// \param value Value as a span of bytes.
    /// \return True if the value was set successfully, false otherwise.
    inline bool set(std::span<const uint8_t> key, std::span<const uint8_t> value)
    {
        return set(key.data(), key.size(), value.data(), value.size());
    }

    /// Get a value from the cache using std::span.
    /// \param key Key as a span of bytes.
    /// \param value Output value as a vector of bytes.
    /// \return True if the value was found and written, false otherwise.
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

    /// Delete a value from the cache using std::span.
    /// \param key Key as a span of bytes.
    /// \return True if the value was deleted successfully, false otherwise.
    inline bool del(std::span<const uint8_t> key) { return del(key.data(), key.size()); }

    /// Evict entries from the cache.
    /// \param max_entries Maximum number of entries to keep.
    /// \param max_size Maximum size of entries to evict.
    /// \return True if entries were evicted successfully, false otherwise.
    bool evict(size_t max_entries, size_t max_size);

    /// Get the current statistics of the cache.
    /// \return Statistics of the cache.
    Stats stats() const;

private:
    Options m_options;

    MDB_env* m_env{nullptr};
    unsigned int m_dbi{0};
    unsigned int m_dbi_meta{0};

    size_t m_max_key_size{0};

    /// Monotonically increasing ticket number for cache entries.
    /// This is used to implement LRU eviction policy.
    std::atomic<uint64_t> m_ticket{0};

    std::atomic<size_t> m_bytes_written{0};

    /// Last error message.
    std::string m_last_error;

    SGL_NON_COPYABLE_AND_MOVABLE(PersistentCache);
};

} // namespace sgl
