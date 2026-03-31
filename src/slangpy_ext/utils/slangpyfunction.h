// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <vector>
#include <map>

#include "nanobind.h"

#include "sgl/core/macros.h"
#include "sgl/core/fwd.h"
#include "sgl/core/object.h"
#include "sgl/core/enum.h"

#include "sgl/device/fwd.h"
#include "sgl/device/resource.h"

#include "utils/slangpy.h"

namespace sgl::slangpy {

enum class FunctionNodeType {
    unknown,
    uniforms,
    kernelgen,
    this_,
    cuda_stream,
    ray_tracing,
};
SGL_ENUM_INFO(
    FunctionNodeType,
    {
        {FunctionNodeType::unknown, "unknown"},
        {FunctionNodeType::uniforms, "uniforms"},
        {FunctionNodeType::kernelgen, "kernelgen"},
        {FunctionNodeType::this_, "this"},
        {FunctionNodeType::cuda_stream, "cuda_stream"},
        {FunctionNodeType::ray_tracing, "ray_tracing"},
    }
);
SGL_ENUM_REGISTER(FunctionNodeType);

class NativeFunctionNode : NativeObject {
    SGL_OBJECT(NativeFunctionNode)
public:
    NativeFunctionNode(NativeFunctionNode* parent, FunctionNodeType type, nb::object data)
        : m_parent(parent)
        , m_type(type)
        , m_data(data)
    {
    }

    void read_signature(SignatureBuilder* builder) const override
    {
        // Delegate to the SignatureBuffer implementation via the builder's buffer.
        read_signature(builder->buffer());
    }

    /// Non-virtual overload for SignatureBuffer (hot path).
    void read_signature(SignatureBuffer& builder) const
    {
        switch (m_type) {
        case sgl::slangpy::FunctionNodeType::uniforms:
        case sgl::slangpy::FunctionNodeType::this_:
            // Uniforms and this don't add to signature.
            break;
        default:
            // Any other type affects kernel so adds to signature.
            NativeObject::read_signature(builder);
            builder << "\n";
            break;
        }
        if (m_parent) {
            m_parent->read_signature(builder);
        }
    }

    /// Fill runtime options by walking the function node chain.
    void gather_runtime_options(NativeCallRuntimeOptions& opts) const
    {
        if (m_parent) {
            m_parent->gather_runtime_options(opts);
        }
        switch (m_type) {
        case sgl::slangpy::FunctionNodeType::this_:
            opts.this_obj = m_data;
            break;
        case sgl::slangpy::FunctionNodeType::uniforms:
            opts.uniforms.push_back(m_data);
            break;
        case sgl::slangpy::FunctionNodeType::cuda_stream:
            opts.cuda_stream = nb::cast<NativeHandle>(m_data);
            break;
        case sgl::slangpy::FunctionNodeType::ray_tracing:
            opts.is_ray_tracing = true;
            break;
        default:
            break;
        }
    }


    NativeFunctionNode* parent() const { return m_parent.get(); }

    FunctionNodeType type() const { return m_type; }

    nb::object data() const { return m_data; }

    NativeFunctionNode* find_root()
    {
        NativeFunctionNode* root = this;
        while (root->parent()) {
            root = root->parent();
        }
        return root;
    }

    /// Get/set the cached call data cache pointer.
    NativeCallDataCache* cache() const { return m_cache.get(); }
    void set_cache(NativeCallDataCache* cache) { m_cache = ref<NativeCallDataCache>(cache); }

    /// Resolve the cache by walking to the root node.
    NativeCallDataCache* resolve_cache()
    {
        NativeFunctionNode* root = find_root();
        return root->cache();
    }

    ref<NativeCallData> build_call_data(NativeCallDataCache* cache, nb::args args, nb::kwargs kwargs);

    /// Core dispatch: resolve/build call data + exec. Used by call() on the fast path.
    nb::object invoke(NativeCallDataCache* cache, nb::args args, nb::kwargs kwargs);

    /// Full call implementation that handles _result type override, _append_to,
    /// error formatting, and delegates to invoke(). Registered as __call__ in nanobind.
    nb::object call(nb::args args, nb::kwargs kwargs);

    /// Common preamble: gather options, prepend this, build signature, resolve/generate call data.
    ref<NativeCallData> resolve_call_data(NativeCallDataCache* cache, nb::args& args, nb::kwargs& kwargs);

    /// Call the backward pass for autograd, caching the bwds CallData on the forward CallData.
    /// This avoids the Python round-trip through function.bwds property.
    /// @param fwds_call_data The forward-pass call data (bwds call data is cached on it).
    /// @param args Positional arguments (containing NativeTorchTensorDiffPair objects).
    /// @param kwargs Keyword arguments (containing NativeTorchTensorDiffPair objects).
    /// @return Result of the backward kernel dispatch.
    nb::object call_bwds(NativeCallData* fwds_call_data, nb::args args, nb::kwargs kwargs);

    void append_to(NativeCallDataCache* cache, CommandEncoder* command_encoder, nb::args args, nb::kwargs kwargs);

    /// Get string representation of the function node.
    std::string to_string() const override;

    virtual ref<NativeCallData> generate_call_data(nb::args args, nb::kwargs kwargs)
    {
        SGL_UNUSED(args);
        SGL_UNUSED(kwargs);
        return nullptr;
    }

    /// Generate the backward-pass call data for autograd.
    /// Called once per forward signature, result is cached on the forward CallData.
    /// The default implementation returns nullptr; Python overrides this to build
    /// a CallData with call_mode=bwds.
    /// @param fwds_call_data The forward-pass call data.
    /// @param args Positional arguments (with DiffPair objects).
    /// @param kwargs Keyword arguments (with DiffPair objects).
    /// @return The backward-pass CallData.
    virtual ref<NativeCallData>
    generate_bwds_call_data(NativeCallData* fwds_call_data, nb::args args, nb::kwargs kwargs)
    {
        SGL_UNUSED(fwds_call_data);
        SGL_UNUSED(args);
        SGL_UNUSED(kwargs);
        return nullptr;
    }

    void garbage_collect()
    {
        m_parent = nullptr;
        m_data = nb::none();
        m_cache = nullptr;
    }

    /// Get or create cached runtime options (avoids heap alloc on repeat calls).
    NativeCallRuntimeOptions& cached_options()
    {
        if (!m_cached_opts)
            m_cached_opts = make_ref<NativeCallRuntimeOptions>();
        m_cached_opts->init();
        return *m_cached_opts;
    }

private:
    ref<NativeFunctionNode> m_parent;
    FunctionNodeType m_type;
    nb::object m_data;
    ref<NativeCallDataCache> m_cache;
    ref<NativeCallRuntimeOptions> m_cached_opts;
};

struct PyNativeFunctionNode : NativeFunctionNode {
    NB_TRAMPOLINE(NativeFunctionNode, 2);
    ref<NativeCallData> generate_call_data(nb::args args, nb::kwargs kwargs) override
    {
        NB_OVERRIDE(generate_call_data, args, kwargs);
    }
    ref<NativeCallData>
    generate_bwds_call_data(NativeCallData* fwds_call_data, nb::args args, nb::kwargs kwargs) override
    {
        NB_OVERRIDE(generate_bwds_call_data, fwds_call_data, args, kwargs);
    }
};

} // namespace sgl::slangpy
