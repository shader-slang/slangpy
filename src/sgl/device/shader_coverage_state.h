// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace sgl {

struct ShaderCoverageState;

/// Owns optional coverage state. Access follows the device's host-side serialization contract.
class ShaderCoverageStateOwner {
public:
    ShaderCoverageStateOwner();
    ~ShaderCoverageStateOwner();

    ShaderCoverageStateOwner(const ShaderCoverageStateOwner&) = delete;
    ShaderCoverageStateOwner& operator=(const ShaderCoverageStateOwner&) = delete;
    ShaderCoverageStateOwner(ShaderCoverageStateOwner&&) = delete;
    ShaderCoverageStateOwner& operator=(ShaderCoverageStateOwner&&) = delete;

    /// Returns the state without allocating.
    ShaderCoverageState* get() const { return m_state.get(); }

    /// Creates state on first use. The caller serializes device mutation.
    ShaderCoverageState& get_or_create();

private:
    std::unique_ptr<ShaderCoverageState> m_state;
};

} // namespace sgl
