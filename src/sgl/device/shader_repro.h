// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/core/object.h"

#include <slang.h>
#include <slang-com-ptr.h>

namespace sgl {

class ReproSession;
class ReproModule;

class SlangRepro : Object {
    SGL_OBJECT(SlangRepro)
public:
    struct Session;
    struct Module;

    using SessionHandle = Session*;
    using ModuleHandle = Module*;

    SlangRepro();
    ~SlangRepro();

    void finish();

    SessionHandle create_session(const slang::SessionDesc& desc);
    void assign(SessionHandle session, slang::ISession* slang_session);

    ModuleHandle load_module(SessionHandle session, const char* module_name);
    ModuleHandle load_module_from_source_string(
        SessionHandle session,
        const char* module_name,
        const char* path,
        const char* source
    );
    void assign(ModuleHandle module, slang::IModule* slang_module);

private:
    struct Impl;
    Impl* m_impl;
};

} // namespace sgl
