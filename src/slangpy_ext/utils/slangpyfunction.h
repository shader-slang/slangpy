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
#include "sgl/device/query.h"

#include "utils/slangpy.h"

namespace sgl::slangpy {

enum class FunctionNodeType { unknown, uniforms, kernelgen, this_, cuda_stream, write_timestamps };
SGL_ENUM_INFO(
    FunctionNodeType,
    {{FunctionNodeType::unknown, "unknown"},
     {FunctionNodeType::uniforms, "uniforms"},
     {FunctionNodeType::kernelgen, "kernelgen"},
     {FunctionNodeType::this_, "this"},
     {FunctionNodeType::cuda_stream, "cuda_stream"},
     {FunctionNodeType::write_timestamps, "write_timestamps"}}
);
SGL_ENUM_REGISTER(FunctionNodeType);

class NativeFunctionNode : NativeObject {
public:
    NativeFunctionNode(NativeFunctionNode* parent, FunctionNodeType type, nb::object data)
        : m_parent(parent)
        , m_type(type)
        , m_data(data)
    {
    }

    void read_signature(SignatureBuilder* builder) const override
    {
        switch (m_type) {
        case sgl::slangpy::FunctionNodeType::uniforms:
        case sgl::slangpy::FunctionNodeType::this_:
            // Uniforms and this don't add to signature.
            break;
        default:
            // Any other type affects kernel so adds to signature.
            NativeObject::read_signature(builder);
            *builder << "\n";
            break;
        }
        if (m_parent) {
            m_parent->read_signature(builder);
        }
    }

    void gather_runtime_options(ref<NativeCallRuntimeOptions> options) const
    {
        if (m_parent) {
            m_parent->gather_runtime_options(options);
        }
        switch (m_type) {
        case sgl::slangpy::FunctionNodeType::this_:
            options->set_this(m_data);
            break;
        case sgl::slangpy::FunctionNodeType::uniforms:
            options->get_uniforms().append(m_data);
            break;
        case sgl::slangpy::FunctionNodeType::cuda_stream:
            options->set_cuda_stream(nb::cast<NativeHandle>(m_data));
            break;
        case sgl::slangpy::FunctionNodeType::write_timestamps: {
            nb::tuple t = nb::cast<nb::tuple>(m_data);
            options->set_query_pool(nb::cast<QueryPool*>(t[0]));
            options->set_query_before_index(nb::cast<uint32_t>(t[1]));
            options->set_query_after_index(nb::cast<uint32_t>(t[2]));
            break;
        }
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

    ref<NativeCallData> build_call_data(NativeCallDataCache* cache, nb::args args, nb::kwargs kwargs);

    nb::object call(NativeCallDataCache* cache, nb::args args, nb::kwargs kwargs);

    void append_to(NativeCallDataCache* cache, CommandEncoder* command_encoder, nb::args args, nb::kwargs kwargs);

    /// Get string representation of the function node.
    std::string to_string() const;

    virtual ref<NativeCallData> generate_call_data(nb::args args, nb::kwargs kwargs)
    {
        SGL_UNUSED(args);
        SGL_UNUSED(kwargs);
        return nullptr;
    }

private:
    ref<NativeFunctionNode> m_parent;
    FunctionNodeType m_type;
    nb::object m_data;
};

struct PyNativeFunctionNode : NativeFunctionNode {
    NB_TRAMPOLINE(NativeFunctionNode, 1);
    ref<NativeCallData> generate_call_data(nb::args args, nb::kwargs kwargs) override
    {
        NB_OVERRIDE(generate_call_data, args, kwargs);
    }
};

} // namespace sgl::slangpy
