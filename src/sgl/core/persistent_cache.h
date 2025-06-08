// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"

#include <filesystem>
#include <span>

// Forward declaration
struct MDB_env;

namespace sgl {

class SGL_API PersistentCache : public Object {
    SGL_OBJECT(PersistentCache)
public:
    using WriteValueFunc = void (*)(const void* data, size_t size, void* user_data);

    PersistentCache(const std::filesystem::path& path);
    ~PersistentCache() override;

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

private:
    MDB_env* m_env{nullptr};
    unsigned int m_dbi{0};

    size_t m_max_key_size{0};

    SGL_NON_COPYABLE_AND_MOVABLE(PersistentCache);
};

} // namespace sgl
