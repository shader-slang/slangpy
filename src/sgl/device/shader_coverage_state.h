// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <atomic>

namespace sgl {

struct ShaderCoverageState;

/// Owns lazily published coverage state until device destruction.
/// Closing the device releases retained resources, but keeps the state and its
/// mutexes alive. Destruction requires that no operation still uses the owner.
class ShaderCoverageStateOwner {
public:
    ShaderCoverageStateOwner() = default;
    ~ShaderCoverageStateOwner();

    ShaderCoverageStateOwner(const ShaderCoverageStateOwner&) = delete;
    ShaderCoverageStateOwner& operator=(const ShaderCoverageStateOwner&) = delete;
    ShaderCoverageStateOwner(ShaderCoverageStateOwner&&) = delete;
    ShaderCoverageStateOwner& operator=(ShaderCoverageStateOwner&&) = delete;

    /// Returns the published state without allocating or locking.
    ShaderCoverageState* get() const { return m_state.load(std::memory_order_acquire); }

    /// Creates state on first use; concurrent callers share the published winner.
    ShaderCoverageState& get_or_create();

private:
    std::atomic<ShaderCoverageState*> m_state{nullptr};
};

} // namespace sgl
