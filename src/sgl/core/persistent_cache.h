// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"

#include <atomic>
#include <filesystem>
#include <span>

// Forward declaration
struct MDB_env;

namespace sgl {

class SGL_API PersistentCache : public Object {
    SGL_OBJECT(PersistentCache)
public:
    using WriteValueFunc = void (*)(const void* data, size_t size, void* user_data);

    struct Options {
        /// Maximum size of the cache on disk.
        size_t max_disk_size{64ull * 1024 * 1024};
        /// Maximum number of entries in the cache.
        size_t max_entries{10000};
        /// Automatically evict entries when the cache is full.
        bool auto_evict{true};
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

    bool open(const std::filesystem::path& path, const Options& options = Options());
    void close();

    bool set(const void* key_data, size_t key_size, const void* value_data, size_t value_size);
    bool get(const void* key_data, size_t key_size, WriteValueFunc write_value_func, void* user_data = nullptr);
    bool del(const void* key_data, size_t key_size);

    inline bool set(std::span<const uint8_t> key, std::span<const uint8_t> value)
    {
        return set(key.data(), key.size(), value.data(), value.size());
    }

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

    inline bool del(std::span<const uint8_t> key) { return del(key.data(), key.size()); }

    bool evict(size_t max_entries, size_t max_size);

    Stats stats() const;

private:
    MDB_env* m_env{nullptr};
    unsigned int m_dbi{0};
    unsigned int m_dbi_meta{0};

    size_t m_max_key_size{0};

    std::atomic<uint64_t> m_ticket{0};

    std::string m_last_error;

    SGL_NON_COPYABLE_AND_MOVABLE(PersistentCache);
};

} // namespace sgl
