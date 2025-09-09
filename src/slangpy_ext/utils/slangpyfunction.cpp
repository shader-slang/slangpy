// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"
#include "sgl/device/command.h"
#include "utils/slangpyfunction.h"
#include <fmt/format.h>

namespace sgl {

template<>
struct GcHelper<slangpy::NativeFunctionNode> {
    void traverse(slangpy::NativeFunctionNode*, GcVisitor& visitor)
    {
        visitor("_native_data");
        visitor("_native_parent");
    }
    void clear(slangpy::NativeFunctionNode* node) { node->garbage_collect(); }
};

} // namespace sgl

namespace sgl::slangpy {


ref<NativeCallData> NativeFunctionNode::build_call_data(NativeCallDataCache* cache, nb::args args, nb::kwargs kwargs)
{
    auto options = make_ref<NativeCallRuntimeOptions>();
    gather_runtime_options(options);

    nb::tuple full_args;
    if (!options->get_this().is_none()) {
        args = nb::cast<nb::args>(nb::make_tuple(options->get_this()) + args);
    }

    auto builder = make_ref<SignatureBuilder>();
    read_signature(builder);
    cache->get_args_signature(builder, args, kwargs);

    std::string sig = builder->str();
    ref<NativeCallData> result = cache->find_call_data(sig);
    if (!result) {
        result = generate_call_data(args, kwargs);
        cache->add_call_data(sig, result);
    }
    return result;
}

nb::object NativeFunctionNode::call(NativeCallDataCache* cache, nb::args args, nb::kwargs kwargs)
{
    auto options = make_ref<NativeCallRuntimeOptions>();
    gather_runtime_options(options);

    nb::tuple full_args;
    if (!options->get_this().is_none()) {
        args = nb::cast<nb::args>(nb::make_tuple(options->get_this()) + args);
    }

    auto builder = make_ref<SignatureBuilder>();
    read_signature(builder);
    cache->get_args_signature(builder, args, kwargs);

    std::string sig = builder->str();
    ref<NativeCallData> call_data = cache->find_call_data(sig);

    if (call_data) {
        if (call_data->is_torch_integration())
            return call_data->_py_torch_call(this, options, args, kwargs);
        else
            return call_data->call(options, args, kwargs);
    } else {
        ref<NativeCallData> new_call_data = generate_call_data(args, kwargs);
        cache->add_call_data(sig, new_call_data);
        if (new_call_data->is_torch_integration())
            return new_call_data->_py_torch_call(this, options, args, kwargs);
        else
            return new_call_data->call(options, args, kwargs);
    }
}

void NativeFunctionNode::append_to(
    NativeCallDataCache* cache,
    CommandEncoder* command_encoder,
    nb::args args,
    nb::kwargs kwargs
)
{
    auto options = make_ref<NativeCallRuntimeOptions>();
    gather_runtime_options(options);

    nb::tuple full_args;
    if (!options->get_this().is_none()) {
        args = nb::cast<nb::args>(nb::make_tuple(options->get_this()) + args);
    }

    auto builder = make_ref<SignatureBuilder>();
    read_signature(builder);
    cache->get_args_signature(builder, args, kwargs);


    std::string sig = builder->str();
    NativeCallData* call_data = cache->find_call_data(sig);

    if (call_data) {
        call_data->append_to(options, command_encoder, args, kwargs);
    } else {
        ref<NativeCallData> new_call_data = generate_call_data(args, kwargs);
        cache->add_call_data(sig, new_call_data);
        new_call_data->append_to(options, command_encoder, args, kwargs);
    }
}

std::string NativeFunctionNode::to_string() const
{
    std::string data_type_name = "None";
    if (!m_data.is_none()) {
        data_type_name = nb::cast<std::string>(m_data.type().attr("__name__"));
    }

    return fmt::format(
        "NativeFunctionNode(\n"
        "  type = {},\n"
        "  parent = {},\n"
        "  data_type = \"{}\"\n"
        ")",
        static_cast<int>(m_type),
        m_parent ? "present" : "None",
        data_type_name
    );
}

} // namespace sgl::slangpy

SGL_PY_EXPORT(utils_slangpy_function)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    nb::sgl_enum<FunctionNodeType>(slangpy, "FunctionNodeType");

    nb::class_<NativeFunctionNode, PyNativeFunctionNode, NativeObject>(slangpy, "NativeFunctionNode")
        .def(
            "__init__",
            [](NativeFunctionNode& self,
               std::optional<NativeFunctionNode*> parent,
               FunctionNodeType type,
               nb::object data) { new (&self) PyNativeFunctionNode(parent.value_or(nullptr), type, data); },
            "parent"_a.none(),
            "type"_a,
            "data"_a.none(),
            D_NA(NativeFunctionNode, NativeFunctionNode)
        )
        .def_prop_ro("_native_parent", &NativeFunctionNode::parent)
        .def_prop_ro("_native_type", &NativeFunctionNode::type)
        .def_prop_ro("_native_data", &NativeFunctionNode::data)
        .def("_find_native_root", &NativeFunctionNode::find_root, D_NA(NativeFunctionNode, find_root))
        .def(
            "_native_build_call_data",
            &NativeFunctionNode::build_call_data,
            "cache"_a,
            "args"_a,
            "kwargs"_a,
            D_NA(NativeFunctionNode, build_call_data)
        )
        .def("_native_call", &NativeFunctionNode::call, "cache"_a, "args"_a, "kwargs"_a, D_NA(NativeFunctionNode, call))
        .def(
            "_native_append_to",
            &NativeFunctionNode::append_to,
            "cache"_a,
            "command_encoder"_a,
            "args"_a,
            "kwargs"_a,
            D_NA(NativeFunctionNode, append_to)
        )
        .def(
            "generate_call_data",
            &NativeFunctionNode::generate_call_data,
            "args"_a,
            "kwargs"_a,
            D_NA(NativeFunctionNode, generate_call_data)
        )
        .def(
            "read_signature",
            &NativeFunctionNode::read_signature,
            "builder"_a,
            D_NA(NativeFunctionNode, read_signature)
        )
        .def(
            "gather_runtime_options",
            &NativeFunctionNode::gather_runtime_options,
            "options"_a,
            D_NA(NativeFunctionNode, gather_runtime_options)
        )
        .def("__repr__", &NativeFunctionNode::to_string);
}
