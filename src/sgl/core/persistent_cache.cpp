// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "persistent_cache.h"

#include <lmdb.h>

#include "sgl/core/error.h"

namespace sgl {

inline int set_last_error(std::string& last_error, const char* expr, int err)
{
    if (err != MDB_SUCCESS) {
        last_error = fmt::format("{} failed with error {} ({})", expr, mdb_strerror(err), err);
    }
    return err;
}

#define MDB_CALL(expr) set_last_error(m_last_error, #expr, (expr))

#define MDB_RETURN_ON_FAIL(expr)                                                                                       \
    do {                                                                                                               \
        if (MDB_CALL(expr) != MDB_SUCCESS)                                                                             \
            return false;                                                                                              \
    } while (0)

// Meta-data struct.
// This is used to store additional information about the cache entries in the "meta" database.
struct MetaData {
    /// Size of the cache entry.
    uint64_t size;
    /// Monotonically increasing ticket number. Used for LRU eviction policy.
    uint64_t ticket;
};

PersistentCache::PersistentCache() { }

PersistentCache::~PersistentCache()
{
    close();
}

bool PersistentCache::open(const std::filesystem::path& path, const Options& options)
{
    m_options = options;

    std::error_code ec;
    if (!std::filesystem::create_directories(path, ec) && ec) {
        m_last_error = fmt::format("Failed to create cache directory ({})", ec.message());
        return false;
    }

    MDB_env* env = nullptr;
    MDB_RETURN_ON_FAIL(mdb_env_create(&env));
    mdb_env_set_flags(env, MDB_NOSYNC, true);
    MDB_RETURN_ON_FAIL(mdb_env_set_maxreaders(env, 126));
    MDB_RETURN_ON_FAIL(mdb_env_set_maxdbs(env, 2));
    MDB_RETURN_ON_FAIL(mdb_env_set_mapsize(env, m_options.max_disk_size));

    m_max_key_size = mdb_env_get_maxkeysize(env);

    MDB_RETURN_ON_FAIL(mdb_env_open(env, path.string().c_str(), 0, 0664));

    MDB_txn* txn = nullptr;
    if (MDB_CALL(mdb_txn_begin(env, nullptr, 0, &txn) != MDB_SUCCESS)) {
        mdb_env_close(env);
        return false;
    }
    if (MDB_CALL(mdb_dbi_open(txn, "data", MDB_CREATE, &m_dbi)) != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        return false;
    }
    if (MDB_CALL(mdb_dbi_open(txn, "meta", MDB_CREATE, &m_dbi_meta)) != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        return false;
    }
    MDB_cursor* cursor;
    if (MDB_CALL(mdb_cursor_open(txn, m_dbi_meta, &cursor)) != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        return false;
    }
    MDB_val key, val;
    uint64_t ticket = 0;
    size_t total_size = 0;
    while (mdb_cursor_get(cursor, &key, &val, MDB_NEXT) == MDB_SUCCESS) {
        if (val.mv_size == sizeof(MetaData)) {
            const MetaData* meta_data = static_cast<const MetaData*>(val.mv_data);
            ticket = std::max(ticket, meta_data->ticket);
            total_size += meta_data->size;
        }
    }
    mdb_cursor_close(cursor);
    if (MDB_CALL(mdb_txn_commit(txn)) != MDB_SUCCESS) {
        mdb_dbi_close(env, m_dbi);
        mdb_dbi_close(env, m_dbi_meta);
        mdb_env_close(env);
        return false;
    }

    // TODO trigger auto-eviction if the cache is too large
    if (m_options.auto_evict) {
    }

    m_env = env;
    m_ticket.store(ticket);

    return true;
}

void PersistentCache::close()
{
    mdb_dbi_close(m_env, m_dbi);
    mdb_dbi_close(m_env, m_dbi_meta);
    mdb_env_close(m_env);
    m_env = nullptr;
}

bool PersistentCache::set(const void* key_data, size_t key_size, const void* value_data, size_t value_size)
{
    SGL_CHECK(m_env != nullptr, "Cache is not open");
    SGL_CHECK(key_size > 0, "Key size must be greater than 0");
    SGL_CHECK(key_size <= m_max_key_size, "Key size exceeds maximum allowed size");

    MDB_txn* txn = nullptr;
    if (MDB_CALL(mdb_txn_begin(m_env, nullptr, 0, &txn)) != MDB_SUCCESS)
        return false;

    MDB_val mdb_key = {key_size, (void*)key_data};
    MDB_val mdb_val = {value_size, (void*)value_data};
    if (MDB_CALL(mdb_put(txn, m_dbi, &mdb_key, &mdb_val, 0) != MDB_SUCCESS)) {
        mdb_txn_abort(txn);
        return false;
    }

    MetaData meta_data{.size = mdb_val.mv_size, .ticket = ++m_ticket};
    MDB_val mdb_val_meta = {sizeof(MetaData), &meta_data};
    if (MDB_CALL(mdb_put(txn, m_dbi_meta, &mdb_key, &mdb_val_meta, 0) != MDB_SUCCESS)) {
        mdb_txn_abort(txn);
        return false;
    }

    if (MDB_CALL(mdb_txn_commit(txn)) != MDB_SUCCESS)
        return false;

    m_bytes_written.fetch_add(value_size, std::memory_order_relaxed);

    if (m_options.auto_evict) {
        size_t bytes_written = m_bytes_written.load(std::memory_order_relaxed);
        while (bytes_written >= m_options.auto_evict_after_bytes) {

        }
    }

    return true;
}

bool PersistentCache::get(const void* key_data, size_t key_size, ValueCallback value_callback, void* user_data)
{
    SGL_CHECK(m_env != nullptr, "Cache is not open");
    SGL_CHECK(key_size > 0, "Key size must be greater than 0");
    SGL_CHECK(key_size <= m_max_key_size, "Key size exceeds maximum allowed size");

    MDB_txn* txn = nullptr;
    if (MDB_CALL(mdb_txn_begin(m_env, nullptr, 0, &txn)) != MDB_SUCCESS)
        return false;

    MDB_val mdb_key = {key_size, (void*)key_data};
    MDB_val mdb_val;

    if (MDB_CALL(mdb_get(txn, m_dbi, &mdb_key, &mdb_val)) != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        return false;
    }

    MetaData meta_data{.size = mdb_val.mv_size, .ticket = ++m_ticket};
    MDB_val mdb_val_meta = {sizeof(MetaData), &meta_data};
    if (MDB_CALL(mdb_put(txn, m_dbi_meta, &mdb_key, &mdb_val_meta, 0) != MDB_SUCCESS)) {
        mdb_txn_abort(txn);
        return false;
    }

    value_callback(mdb_val.mv_data, mdb_val.mv_size, user_data);

    return MDB_CALL(mdb_txn_commit(txn)) == MDB_SUCCESS;
}

bool PersistentCache::del(const void* key_data, size_t key_size)
{
    SGL_CHECK(m_env != nullptr, "Cache is not open");
    SGL_CHECK(key_size > 0, "Key size must be greater than 0");
    SGL_CHECK(key_size <= m_max_key_size, "Key size exceeds maximum allowed size");

    MDB_txn* txn = nullptr;
    if (MDB_CALL(mdb_txn_begin(m_env, nullptr, 0, &txn)) != MDB_SUCCESS)
        return false;

    MDB_val mdb_key = {key_size, (void*)key_data};

    if (MDB_CALL(mdb_del(txn, m_dbi, &mdb_key, nullptr) != MDB_SUCCESS)) {
        mdb_txn_abort(txn);
        return false;
    }

    if (MDB_CALL(mdb_del(txn, m_dbi_meta, &mdb_key, nullptr) != MDB_SUCCESS)) {
        mdb_txn_abort(txn);
        return false;
    }

    return MDB_CALL(mdb_txn_commit(txn)) == MDB_SUCCESS;
}

bool PersistentCache::evict(size_t max_entries, size_t max_size)
{
    SGL_CHECK(m_env != nullptr, "Cache is not open");

    MDB_txn* txn = nullptr;
    if (MDB_CALL(mdb_txn_begin(m_env, nullptr, 0, &txn)) != MDB_SUCCESS)
        return false;

    MDB_cursor* cursor;
    if (MDB_CALL(mdb_cursor_open(txn, m_dbi_meta, &cursor)) != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        return false;
    }

    struct Entry {
        MetaData meta_data;
        char key[512];
        size_t key_size;
    };
    std::vector<Entry> entries;
    entries.reserve(1000);

    size_t total_size = 0;

    MDB_val key, val;
    while (mdb_cursor_get(cursor, &key, &val, MDB_NEXT) == MDB_SUCCESS) {
        if (val.mv_size == sizeof(MetaData)) {
            Entry& entry = entries.emplace_back();
            entry.meta_data = *static_cast<const MetaData*>(val.mv_data);
            std::memcpy(entry.key, key.mv_data, key.mv_size);
            entry.key_size = key.mv_size;
            total_size += entry.meta_data.size;
        }
    }

    std::sort(
        entries.begin(),
        entries.end(),
        [](const Entry& a, const Entry& b) { return a.meta_data.ticket < b.meta_data.ticket; }
    );

    size_t total_entries = entries.size();
    size_t evicted_entries = 0;
    size_t evicted_size = 0;

    for (size_t i = 0; i < entries.size(); ++i) {
        if (total_entries <= max_entries && total_size <= max_size)
            break;

        const Entry& entry = entries[i];
        MDB_val mdb_key = {entry.key_size, (void*)entry.key};
        if (MDB_CALL(mdb_del(txn, m_dbi, &mdb_key, nullptr) != MDB_SUCCESS)) {
            mdb_cursor_close(cursor);
            mdb_txn_abort(txn);
            return false;
        }
        if (MDB_CALL(mdb_del(txn, m_dbi_meta, &mdb_key, nullptr) != MDB_SUCCESS)) {
            mdb_cursor_close(cursor);
            mdb_txn_abort(txn);
            return false;
        }

        total_entries--;
        total_size -= entry.meta_data.size;
        evicted_entries++;
        evicted_size += entry.meta_data.size;
    }

    mdb_cursor_close(cursor);
    return MDB_CALL(mdb_txn_commit(txn)) == MDB_SUCCESS;
}

PersistentCache::Stats PersistentCache::stats() const
{
    SGL_CHECK(m_env != nullptr, "Cache is not open");

    Stats stats;

    MDB_txn* txn = nullptr;
    if (mdb_txn_begin(m_env, nullptr, MDB_RDONLY, &txn) != MDB_SUCCESS)
        return stats;

    MDB_cursor* cursor;
    if (mdb_cursor_open(txn, m_dbi_meta, &cursor) != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        return stats;
    }

    MDB_val key, val;
    while (mdb_cursor_get(cursor, &key, &val, MDB_NEXT) == MDB_SUCCESS) {
        if (val.mv_size == sizeof(MetaData)) {
            const MetaData* meta_data = static_cast<const MetaData*>(val.mv_data);
            stats.entries++;
            stats.size += meta_data->size;
        }
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);

    return stats;
}

} // namespace sgl
