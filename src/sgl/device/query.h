// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"
#include "sgl/device/device_child.h"

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/core/enum.h"

#include <slang-rhi.h>

#include <span>

namespace sgl {

struct QueryPoolDesc {
    /// Query type.
    QueryType type;
    /// Number of queries in the pool.
    uint32_t count;
};

class SGL_API QueryPool : public DeviceChild {
    SGL_OBJECT(QueryPool)
public:
    QueryPool(ref<Device> device, QueryPoolDesc desc);

    virtual void _release_rhi_resources() override { m_rhi_query_pool.setNull(); }

    const QueryPoolDesc& desc() const { return m_desc; }

    void reset();

    void get_results(uint32_t index, uint32_t count, std::span<uint64_t> result);
    std::vector<uint64_t> get_results(uint32_t index, uint32_t count);
    uint64_t get_result(uint32_t index);

    void get_timestamp_results(uint32_t index, uint32_t count, std::span<double> result);
    std::vector<double> get_timestamp_results(uint32_t index, uint32_t count);
    double get_timestamp_result(uint32_t index);

    rhi::IQueryPool* rhi_query_pool() const { return m_rhi_query_pool; }

    std::string to_string() const override;

private:
    QueryPoolDesc m_desc;
    Slang::ComPtr<rhi::IQueryPool> m_rhi_query_pool;
};

} // namespace sgl
