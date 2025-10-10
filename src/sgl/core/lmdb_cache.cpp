// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "lmdb_cache.h"

#include "sgl/core/error.h"
#include "sgl/core/platform.h"

#include <chrono>

#include <lmdb.h>

// Brief overview on how the cache works:
// - The cache is backed by an LMDB database stored on disk.
// - There are two databases:
//   - "data": stores the actual key-value pairs.
//   - "meta": stores meta-data for each entry (currently only last access time).
// - Each entry in the "data" database is identified by its key.
// - The "meta" database uses the same keys as the "data" database to store the corresponding meta-data.
// - When setting a value, we also update the last access time in the "meta" database.
// - When getting a value, we update the last access time in the "meta" database.
// - When deleting a value, we also delete the corresponding meta-data entry.
// - Eviction is triggered when the cache size exceeds a certain threshold (eviction_threshold),
//   and we evict entries until the cache size is below a target size (eviction_target).
// - Eviction is done by scanning all entries in the "meta" database, sorting them by last access time,
//   and deleting the least recently used entries until we are below the target size.
//
// Possible improvements:
// - Instead of blocking on eviction, we could run it in a background thread.
// - Instead of scanning all entries during eviction, we could sample a subset of entries
//   and evict the least recently used among them. This would reduce the eviction time
//   at the cost of not evicting the absolute least recently used entries.

#define LMDB_THROW(msg, error) throw sgl::LMDBException(fmt::format("{} ({})", msg, mdb_strerror(error)), error)

namespace sgl {

class ScopedTransaction {
public:
    ScopedTransaction(MDB_env* env, unsigned int flags = 0)
    {
        if (int result = mdb_txn_begin(env, nullptr, flags, &m_txn); result != MDB_SUCCESS)
            LMDB_THROW("Failed to begin transaction", result);
    }

    ~ScopedTransaction()
    {
        if (m_txn)
            mdb_txn_abort(m_txn);
    }

    void commit()
    {
        SGL_CHECK(m_txn != nullptr, "Transaction is already committed or aborted");
        // It's an error to call `mdb_txn_abort` after `mdb_txn_commit`, even if it fails.
        MDB_txn* txn = m_txn;
        m_txn = nullptr;
        if (int result = mdb_txn_commit(txn); result != MDB_SUCCESS)
            LMDB_THROW("Failed to commit transaction", result);
    }

    operator MDB_txn*() { return m_txn; }

private:
    MDB_txn* m_txn;
};

class ScopedCursor {
public:
    ScopedCursor(MDB_txn* txn, unsigned int dbi)
        : m_cursor(nullptr)
    {
        if (int result = mdb_cursor_open(txn, dbi, &m_cursor); result != MDB_SUCCESS)
            LMDB_THROW("Failed to open cursor", result);
    }

    ~ScopedCursor() { close(); }

    void close()
    {
        if (m_cursor)
            mdb_cursor_close(m_cursor);
        m_cursor = nullptr;
    }

    operator MDB_cursor*() { return m_cursor; }

private:
    MDB_cursor* m_cursor;
};

// Meta-data struct.
// This is used to store additional information about the cache entries in the "meta" database.
struct MetaData {
    /// Time of last access (in nanoseconds since epoch).
    uint64_t last_access;
};

inline uint64_t get_current_time_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}

LMDBCache::LMDBCache(const std::filesystem::path& path, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});

    std::error_code ec;
    if (!std::filesystem::create_directories(path, ec) && ec)
        SGL_THROW("Failed to create cache directory ({})", ec.message());

    m_db = open_db(path, options);

    m_max_key_size = mdb_env_get_maxkeysize(m_db.env);

    m_eviction_threshold_size = (options.eviction_threshold * options.max_size) / 100;
    m_eviction_target_size = (options.eviction_target * options.max_size) / 100;
}

LMDBCache::~LMDBCache()
{
    close_db(m_db);
}

void LMDBCache::set(const void* key_data, size_t key_size, const void* value_data, size_t value_size)
{
    SGL_CHECK(key_size > 0, "Key size must be greater than 0");
    SGL_CHECK(key_size <= m_max_key_size, "Key size exceeds maximum allowed size");

    // Trigger eviction if necessary.
    if (usage().used_size > m_eviction_threshold_size)
        evict();

    ScopedTransaction txn(m_db.env);

    MDB_val mdb_key = {key_size, const_cast<void*>(key_data)};
    MDB_val mdb_val = {value_size, const_cast<void*>(value_data)};
    if (int result = mdb_put(txn, m_db.dbi_data, &mdb_key, &mdb_val, 0); result != MDB_SUCCESS)
        LMDB_THROW("Failed to write data", result);

    MetaData meta_data{.last_access = get_current_time_ns()};
    MDB_val mdb_val_meta = {sizeof(MetaData), &meta_data};
    if (int result = mdb_put(txn, m_db.dbi_meta, &mdb_key, &mdb_val_meta, 0); result != MDB_SUCCESS)
        LMDB_THROW("Failed to write metadata", result);

    txn.commit();
}

bool LMDBCache::get(const void* key_data, size_t key_size, WriteValueFunc write_value_func, void* user_data)
{
    SGL_CHECK(key_size > 0, "Key size must be greater than 0");
    SGL_CHECK(key_size <= m_max_key_size, "Key size exceeds maximum allowed size");

    ScopedTransaction txn(m_db.env);

    MDB_val mdb_key = {key_size, const_cast<void*>(key_data)};
    MDB_val mdb_val;

    int result = mdb_get(txn, m_db.dbi_data, &mdb_key, &mdb_val);
    if (result == MDB_NOTFOUND)
        return false;
    if (result != MDB_SUCCESS)
        LMDB_THROW("Failed to read data", result);

    MetaData meta_data{.last_access = get_current_time_ns()};
    MDB_val mdb_val_meta = {sizeof(MetaData), &meta_data};
    result = mdb_put(txn, m_db.dbi_meta, &mdb_key, &mdb_val_meta, 0);
    if (result != MDB_SUCCESS)
        LMDB_THROW("Failed to write metadata", result);

    write_value_func(mdb_val.mv_data, mdb_val.mv_size, user_data);

    txn.commit();

    return true;
}

bool LMDBCache::del(const void* key_data, size_t key_size)
{
    SGL_CHECK(key_size > 0, "Key size must be greater than 0");
    SGL_CHECK(key_size <= m_max_key_size, "Key size exceeds maximum allowed size");

    ScopedTransaction txn(m_db.env);

    MDB_val mdb_key = {key_size, const_cast<void*>(key_data)};

    int result = mdb_del(txn, m_db.dbi_data, &mdb_key, nullptr);
    if (result == MDB_NOTFOUND)
        return false;
    if (result != MDB_SUCCESS)
        LMDB_THROW("Failed to delete data", result);

    result = mdb_del(txn, m_db.dbi_meta, &mdb_key, nullptr);
    if (result != MDB_SUCCESS && result != MDB_NOTFOUND)
        LMDB_THROW("Failed to delete metadata", result);

    txn.commit();

    return true;
}

LMDBCache::Usage LMDBCache::usage() const
{
    Usage usage;

    ScopedTransaction txn(m_db.env, MDB_RDONLY);

    uint64_t used_pages = 0;
    uint64_t page_size = 0;
    for (MDB_dbi dbi : {MDB_dbi(0) /* FREE_DBI */, MDB_dbi(1) /* MAIN_DBI */, m_db.dbi_data, m_db.dbi_meta}) {
        MDB_stat stat = {};
        if (int result = mdb_stat(txn, dbi, &stat); result != MDB_SUCCESS)
            LMDB_THROW("Failed to get DB stats", result);
        used_pages += (stat.ms_branch_pages + stat.ms_leaf_pages + stat.ms_overflow_pages);
        page_size = stat.ms_psize;
    }

    MDB_envinfo info = {};
    if (int result = mdb_env_info(m_db.env, &info); result != MDB_SUCCESS)
        LMDB_THROW("Failed to get environment info", result);

    usage.reserved_size = info.me_mapsize;
    usage.committed_size = (info.me_last_pgno + 1) * page_size;
    usage.used_size = used_pages * page_size;

    return usage;
}

LMDBCache::Stats LMDBCache::stats() const
{
    Stats stats;

    stats.evictions = m_evictions.load();

    ScopedTransaction txn(m_db.env, MDB_RDONLY);
    ScopedCursor cursor(txn, m_db.dbi_data);

    MDB_val key, val;
    while (mdb_cursor_get(cursor, &key, &val, MDB_NEXT) == MDB_SUCCESS) {
        stats.entries++;
        stats.size += val.mv_size;
    }

    return stats;
}

void LMDBCache::evict()
{
    struct Entry {
        uint64_t last_access;
        MDB_val key;
    };
    std::vector<Entry> entries;

    size_t used_size = usage().used_size;
    // fmt::println("Evicting entries: used_size={} target_size={}", used_size, m_eviction_target_size);
    if (used_size < m_eviction_target_size)
        return;
    size_t required_free_size = used_size - m_eviction_target_size;

    ScopedTransaction txn(m_db.env);
    ScopedCursor cursor(txn, m_db.dbi_meta);

    // Scan all entries.
    MDB_val key, val;
    while (mdb_cursor_get(cursor, &key, &val, MDB_NEXT) == MDB_SUCCESS) {
        entries.push_back({
            .last_access = static_cast<const MetaData*>(val.mv_data)->last_access,
            .key = key,
        });
    }

    // Create heap based on last access time (oldest first).
    auto cmp = [](const Entry& a, const Entry& b) { return a.last_access > b.last_access; };
    std::make_heap(entries.begin(), entries.end(), cmp);

    // Evict entries until we are below the target size.
    size_t evictions = 0;
    while (required_free_size > 0 && !entries.empty()) {
        std::pop_heap(entries.begin(), entries.end(), cmp);
        Entry& entry = entries.back();
        if (int result = mdb_get(txn, m_db.dbi_data, &entry.key, &val); result != MDB_SUCCESS)
            LMDB_THROW("Failed to get data during eviction", result);
        required_free_size -= std::min(required_free_size, val.mv_size);
        if (int result = mdb_del(txn, m_db.dbi_data, &entry.key, nullptr); result != MDB_SUCCESS)
            LMDB_THROW("Failed to delete data during eviction", result);
        if (int result = mdb_del(txn, m_db.dbi_meta, &entry.key, nullptr); result != MDB_SUCCESS)
            LMDB_THROW("Failed to delete metadata during eviction", result);
        entries.pop_back();
        evictions++;
    }

    cursor.close();
    txn.commit();

    // fmt::println("Eviction complete: used_size={} evictions={}", usage().used_size, evictions);

    m_evictions.fetch_add(evictions);
}

// LMDB doesn't support opening the same DB environment multiple times in the same process.
// To work around this, we keep a global list of open environments to reuse them if opened multiple times.

struct DBCacheItem {
    uint64_t ref_count;
    ProcessID pid;
    std::filesystem::path path;
    LMDBCache::DB db;
};

std::vector<DBCacheItem> s_db_cache;
std::mutex s_db_cache_mutex;

LMDBCache::DB LMDBCache::open_db(const std::filesystem::path& path, const Options& options)
{
    ProcessID pid = platform::current_process_id();
    std::filesystem::path abs_path = std::filesystem::absolute(path);
    std::lock_guard lock(s_db_cache_mutex);
    auto it = std::find_if(
        s_db_cache.begin(),
        s_db_cache.end(),
        [pid, &abs_path](const DBCacheItem& e) { return e.pid == pid && e.path == abs_path; }
    );
    if (it != s_db_cache.end()) {
        it->ref_count++;
        return it->db;
    }

    DB db = {};

    if (int result = mdb_env_create(&db.env); result != MDB_SUCCESS)
        LMDB_THROW("Failed to create environment", result);
    if (int result = mdb_env_set_maxreaders(db.env, 126); result != MDB_SUCCESS)
        LMDB_THROW("Failed to set max readers", result);
    if (int result = mdb_env_set_maxdbs(db.env, 2); result != MDB_SUCCESS)
        LMDB_THROW("Failed to set max DBs", result);
    if (int result = mdb_env_set_mapsize(db.env, options.max_size); result != MDB_SUCCESS)
        LMDB_THROW("Failed to set map size", result);

    int flags = options.nosync ? MDB_NOSYNC : 0;
    if (int result = mdb_env_open(db.env, abs_path.string().c_str(), flags, 0664); result != MDB_SUCCESS)
        LMDB_THROW("Failed to open environment", result);

    ScopedTransaction txn(db.env);

    if (int result = mdb_dbi_open(txn, "data", MDB_CREATE, &db.dbi_data); result != MDB_SUCCESS)
        LMDB_THROW("Failed to open data DB", result);
    if (int result = mdb_dbi_open(txn, "meta", MDB_CREATE, &db.dbi_meta); result != MDB_SUCCESS)
        LMDB_THROW("Failed to open meta DB", result);

    txn.commit();

    s_db_cache.push_back(
        DBCacheItem{
            .ref_count = 1,
            .pid = pid,
            .path = abs_path,
            .db = db,
        }
    );

    return db;
}

void LMDBCache::close_db(DB db)
{
    ProcessID pid = platform::current_process_id();
    std::lock_guard lock(s_db_cache_mutex);
    auto it = std::find_if(
        s_db_cache.begin(),
        s_db_cache.end(),
        [&db](const DBCacheItem& item) { return item.db.env == db.env; }
    );
    SGL_ASSERT(it != s_db_cache.end());
    SGL_ASSERT(it->pid == pid);
    if (--it->ref_count == 0) {
        mdb_dbi_close(db.env, db.dbi_data);
        mdb_dbi_close(db.env, db.dbi_meta);
        mdb_env_close(db.env);
        s_db_cache.erase(it);
    }
}

} // namespace sgl
