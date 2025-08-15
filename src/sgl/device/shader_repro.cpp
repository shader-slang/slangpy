// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shader_repro.h"

#include "sgl/core/logger.h"
#include "sgl/core/format.h"

namespace sgl {

static const std::string REPRO_HEADER = R"(
#include <slang.h>
#include <slang-com-ptr.h>

template<typename T>
using ComPtr = Slang::ComPtr<T>;

int main()
{
    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_RETURN_ON_FAIL(slang::createGlobalSession(globalSession.writeRef()));

)";

static const std::string REPRO_FOOTER = R"(
    return SLANG_OK;
}
)";

class CPPWriter {
public:
    void indent()
    {
        m_indent += 4;
        m_indent_str = std::string(m_indent, ' ');
    }

    void unindent()
    {
        m_indent = std::max(0u, m_indent - 4);
        m_indent_str = std::string(m_indent, ' ');
    }

    void write_raw(std::string_view code) { m_code += code; }

    void write(const std::string_view str)
    {
        write_raw(m_indent_str);
        write_raw(str);
    }

    template<typename... Args>
    inline void write(fmt::format_string<Args...> fmt, Args&&... args)
    {
        write(fmt::format(fmt, std::forward<Args>(args)...));
    }

    void writeln(const std::string_view str)
    {
        write_raw(m_indent_str);
        write_raw(str);
        write_raw("\n");
    }

    template<typename... Args>
    inline void writeln(fmt::format_string<Args...> fmt, Args&&... args)
    {
        writeln(fmt::format(fmt, std::forward<Args>(args)...));
    }

    const std::string& code() const { return m_code; }


private:
    std::string m_code;
    uint32_t m_indent{0};
    std::string m_indent_str;
};

#define WRITE(...) m_impl->writer.write(__VA_ARGS__)
#define WRITELN(...) m_impl->writer.writeln(__VA_ARGS__)

struct SlangRepro::Session {
    Slang::ComPtr<slang::ISession> slang_session;
    std::string identifier;
};

struct SlangRepro::Module {
    Slang::ComPtr<slang::IModule> slang_module;
    std::string identifier;
};

struct SlangRepro::Impl {
    CPPWriter writer;
    std::vector<Session*> sessions;
    std::vector<Module*> modules;
};

SlangRepro::SlangRepro()
{
    m_impl = new Impl();
    m_impl->writer.write_raw(REPRO_HEADER);
    m_impl->writer.indent();
}

SlangRepro::~SlangRepro()
{
    finish();
    delete m_impl;
}

void SlangRepro::finish()
{
    m_impl->writer.unindent();
    m_impl->writer.write_raw(REPRO_FOOTER);
    printf("%s", m_impl->writer.code().c_str());
}

SlangRepro::SessionHandle SlangRepro::create_session(const slang::SessionDesc& desc)
{
    SLANG_UNUSED(desc);
    Session* session = new Session();
    session->identifier = fmt::format("session{}", m_impl->sessions.size());
    m_impl->sessions.push_back(session);

    // clang-format off
    WRITELN("slang::SessionDesc {}Desc;", session->identifier);
    WRITELN("ComPtr<slang::ISession> {};", session->identifier);
    WRITELN("SLANG_RETURN_ON_FAIL(globalSession->createSession({}Desc, {}.writeRef()));", session->identifier, session->identifier);
    // clang-format on

    return session;
}

void SlangRepro::assign(SessionHandle session, slang::ISession* slang_session)
{
    session->slang_session = slang_session;
}

SlangRepro::ModuleHandle SlangRepro::load_module(SessionHandle session, const char* module_name)
{
    SLANG_UNUSED(session);
    SLANG_UNUSED(module_name);
    Module* module = new Module();
    m_impl->modules.push_back(module);
    return module;
}

SlangRepro::ModuleHandle SlangRepro::load_module_from_source_string(
    SessionHandle session,
    const char* module_name,
    const char* path,
    const char* source
)
{
    SLANG_UNUSED(session);
    SLANG_UNUSED(module_name);
    SLANG_UNUSED(path);
    SLANG_UNUSED(source);
    Module* module = new Module();
    m_impl->modules.push_back(module);
    return module;
}

void SlangRepro::assign(ModuleHandle module, slang::IModule* slang_module)
{
    module->slang_module = slang_module;
}

} // namespace sgl
