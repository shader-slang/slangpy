// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "persistent_cache.h"

#include <lmdb.h>

#include "sgl/core/error.h"

namespace sgl {

#define MDB_THROW_ON_FAIL(expr)                                                                                        \
    {                                                                                                                  \
        int _mdb_err = (expr);                                                                                         \
        if (_mdb_err != MDB_SUCCESS)                                                                                   \
            SGL_THROW("{} failed with error {} ({})", #expr, mdb_strerror(_mdb_err), _mdb_err);                        \
    }

PersistentCache::PersistentCache(const std::filesystem::path& path)
{
    std::filesystem::create_directories(path);

    MDB_THROW_ON_FAIL(mdb_env_create(&m_env));
    MDB_THROW_ON_FAIL(mdb_env_set_maxreaders(m_env, 126));
    MDB_THROW_ON_FAIL(mdb_env_set_maxdbs(m_env, 1));
    MDB_THROW_ON_FAIL(mdb_env_set_mapsize(m_env, 64ULL * 1024 * 1024));

    m_max_key_size = mdb_env_get_maxkeysize(m_env);

    MDB_THROW_ON_FAIL(mdb_env_open(m_env, path.string().c_str(), 0, 0664));

    MDB_txn* txn = nullptr;
    MDB_THROW_ON_FAIL(mdb_txn_begin(m_env, nullptr, 0, &txn));
    MDB_THROW_ON_FAIL(mdb_dbi_open(txn, nullptr, MDB_CREATE, &m_dbi));
    MDB_THROW_ON_FAIL(mdb_txn_commit(txn));
}

PersistentCache::~PersistentCache()
{
    mdb_dbi_close(m_env, m_dbi);
    mdb_env_close(m_env);
}

bool PersistentCache::set(const void* key_data, size_t key_size, const void* value_data, size_t value_size)
{
    SGL_UNUSED(key_data, key_size, value_data, value_size);
    return true;
#if 0
    SGL_CHECK(key_size > 0, "Key size must be greater than 0");
    SGL_CHECK(key_size <= m_max_key_size, "Key size exceeds maximum allowed size");

    MDB_txn* txn = nullptr;
    if (mdb_txn_begin(m_env, nullptr, 0, &txn) != MDB_SUCCESS)
        return false;

    MDB_val mdb_key = {key_size, (void*)key_data};
    MDB_val mdb_val = {value_size, (void*)value_data};

    int rc = mdb_put(txn, m_dbi, &mdb_key, &mdb_val, 0);
    if (rc == 0)
        mdb_txn_commit(txn);
    else
        mdb_txn_abort(txn);

    return rc == 0;
#endif
}

bool PersistentCache::get(const void* key_data, size_t key_size, WriteValueFunc write_value_func, void* user_data)
{
    SGL_CHECK(key_size > 0, "Key size must be greater than 0");
    SGL_CHECK(key_size <= m_max_key_size, "Key size exceeds maximum allowed size");

    MDB_txn* txn = nullptr;
    if (mdb_txn_begin(m_env, nullptr, MDB_RDONLY, &txn) != MDB_SUCCESS)
        return false;

    MDB_val mdb_key = {key_size, (void*)key_data};
    MDB_val mdb_val;

    int rc = mdb_get(txn, m_dbi, &mdb_key, &mdb_val);
    if (rc == 0)
        write_value_func(mdb_val.mv_data, mdb_val.mv_size, user_data);
    mdb_txn_abort(txn);
    return rc == 0;
}

bool PersistentCache::del(const void* key_data, size_t key_size)
{
    SGL_CHECK(key_size > 0, "Key size must be greater than 0");
    SGL_CHECK(key_size <= m_max_key_size, "Key size exceeds maximum allowed size");

    MDB_txn* txn = nullptr;
    if (mdb_txn_begin(m_env, nullptr, 0, &txn) != MDB_SUCCESS)
        return false;

    MDB_val mdb_key = {key_size, (void*)key_data};
    int rc = mdb_del(txn, m_dbi, &mdb_key, nullptr);
    if (rc == 0)
        mdb_txn_commit(txn);
    else
        mdb_txn_abort(txn);

    return rc == 0;
}


} // namespace sgl
