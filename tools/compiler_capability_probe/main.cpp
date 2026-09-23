// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Deliberately uses createSession(), not the CLI's additional profile validation.
// Run each experiment in a separate process so compiler libraries cannot mix.
#include <slang.h>
#include <slang-com-ptr.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static std::string json_string(const std::string& value)
{
    std::ostringstream result;
    result << '"';
    const char* hex = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\')
            result << '\\' << c;
        else if (c < 32)
            result << "\\u00" << hex[c >> 4] << hex[c & 15];
        else
            result << c;
    }
    result << '"';
    return result.str();
}

static std::string read_file(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot read " + path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

static slang::CompilerOptionEntry int_option(slang::CompilerOptionName name, int value)
{
    slang::CompilerOptionEntry entry{};
    entry.name = name;
    entry.value.kind = slang::CompilerOptionValueKind::Int;
    entry.value.intValue0 = value;
    return entry;
}

int main(int argc, char** argv)
{
    std::map<std::string, std::string> report{{"status", "error"}, {"stage", "arguments"}};
    Slang::ComPtr<slang::IGlobalSession> global;
    int exit_code = 1;
    try {
        std::map<std::string, std::vector<std::string>> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc)
                throw std::runtime_error("Options must be --name value pairs");
            args[argv[i]].push_back(argv[i + 1]);
        }
        auto get = [&](const std::string& name, const std::string& fallback = "") -> std::string
        {
            return args.count(name) ? args.at(name).back() : fallback;
        };

        report["stage"] = "load_library";
        auto library = std::filesystem::absolute(get("--library"));
        using CreateSession = SlangResult (*)(SlangInt, slang::IGlobalSession**);
#ifdef _WIN32
        SetDllDirectoryW(library.parent_path().c_str());
        auto handle = LoadLibraryW(library.c_str());
        if (!handle)
            throw std::runtime_error("LoadLibrary failed: " + std::to_string(GetLastError()));
        auto create = reinterpret_cast<CreateSession>(GetProcAddress(handle, "slang_createGlobalSession"));
        wchar_t loaded_path[32768];
        auto count = GetModuleFileNameW(handle, loaded_path, 32768);
        report["library"] = std::filesystem::path(std::wstring(loaded_path, count)).string();
#else
        auto handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle)
            throw std::runtime_error(dlerror());
        auto create = reinterpret_cast<CreateSession>(dlsym(handle, "slang_createGlobalSession"));
        report["library"] = library.string();
#endif
        // Keep the library loaded until process exit; COM objects must be destroyed first.
        if (!create)
            throw std::runtime_error("Missing slang_createGlobalSession export");
        if (SLANG_FAILED(create(SLANG_API_VERSION, global.writeRef())))
            throw std::runtime_error("Cannot create global session");
        report["build_tag"] = global->getBuildTagString();
        auto compiler_directory = [&](const std::string& option)
        {
            std::filesystem::path path(get(option));
            return std::filesystem::is_regular_file(path) ? path.parent_path().string() : path.string();
        };
        if (!get("--dxc").empty())
            global->setDownstreamCompilerPath(SLANG_PASS_THROUGH_DXC, compiler_directory("--dxc").c_str());
        if (!get("--nvrtc").empty()) {
            // The default Slang loader appends the platform suffix to the NVRTC basename.
            // Unlike DXC's locator, this locator expects a library name, not a directory.
            std::filesystem::path path(get("--nvrtc"));
#ifdef _WIN32
            if (path.extension() == ".dll")
                path.replace_extension();
#endif
            global->setDownstreamCompilerPath(SLANG_PASS_THROUGH_NVRTC, path.string().c_str());
        }

        std::vector<slang::CompilerOptionEntry> options;
        report["stage"] = "resolve_options";
        for (const auto& capability : args["--capability"]) {
            auto id = global->findCapability(capability.c_str());
            report["lookup:" + capability] = std::to_string(int(id));
            if (id == SLANG_CAPABILITY_UNKNOWN) {
                if (get("--ignore-unknown") == "1")
                    continue;
                throw std::runtime_error("Unknown capability: " + capability);
            }
            options.push_back(int_option(slang::CompilerOptionName::Capability, int(id)));
        }
        for (const auto& capability : args["--lookup"])
            report["lookup:" + capability] = std::to_string(int(global->findCapability(capability.c_str())));
        options.push_back(int_option(slang::CompilerOptionName::RestrictiveCapabilityCheck, get("--strict") == "1"));
        options.push_back(int_option(slang::CompilerOptionName::Optimization, SLANG_OPTIMIZATION_LEVEL_NONE));
        const std::string compiler = get("--target") == "ptx" ? "nvrtc" : "dxc";
        for (const auto& argument : args["--downstream-arg"]) {
            slang::CompilerOptionEntry entry{};
            entry.name = slang::CompilerOptionName::DownstreamArgs;
            entry.value.kind = slang::CompilerOptionValueKind::String;
            entry.value.stringValue0 = compiler.c_str();
            entry.value.stringValue1 = argument.c_str();
            options.push_back(entry);
        }
        const std::map<std::string, SlangCompileTarget> formats{
            {"hlsl", SLANG_HLSL},
            {"dxil", SLANG_DXIL},
            {"dxil-asm", SLANG_DXIL_ASM},
            {"spirv", SLANG_SPIRV},
            {"spirv-asm", SLANG_SPIRV_ASM},
            {"ptx", SLANG_PTX},
            {"cuda", SLANG_CUDA_SOURCE},
            {"metal", SLANG_METAL},
            {"wgsl", SLANG_WGSL},
            {"cpp", SLANG_CPP_SOURCE},
        };
        slang::TargetDesc target{};
        target.format = formats.at(get("--target", "hlsl"));
        target.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
        target.forceGLSLScalarBufferLayout = true;
        if (get("--whole-program") == "1")
            target.flags |= SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM;
        if (!get("--profile").empty()) {
            target.profile = global->findProfile(get("--profile").c_str());
            if (target.profile == SLANG_PROFILE_UNKNOWN)
                throw std::runtime_error("Unknown profile: " + get("--profile"));
        }
        std::vector<const char*> includes;
        for (const auto& path : args["--include"])
            includes.push_back(path.c_str());
        slang::SessionDesc desc{};
        desc.targets = &target;
        desc.targetCount = 1;
        desc.searchPaths = includes.data();
        desc.searchPathCount = includes.size();
        desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
        if (get("--option-scope") == "session") {
            desc.compilerOptionEntries = options.data();
            desc.compilerOptionEntryCount = uint32_t(options.size());
        } else {
            target.compilerOptionEntries = options.data();
            target.compilerOptionEntryCount = uint32_t(options.size());
        }
        Slang::ComPtr<slang::ISession> session;
        report["stage"] = "create_session";
        if (SLANG_FAILED(global->createSession(desc, session.writeRef())))
            throw std::runtime_error("Cannot create session");

        auto diagnose = [&](ISlangBlob* blob)
        {
            if (blob)
                report["diagnostics"]
                    += std::string(static_cast<const char*>(blob->getBufferPointer()), blob->getBufferSize());
        };
        auto check = [&](SlangResult result, ISlangBlob* diagnostics)
        {
            diagnose(diagnostics);
            if (SLANG_FAILED(result))
                throw std::runtime_error("Slang returned " + std::to_string(result));
        };
        if (!get("--source").empty()) {
            auto source = read_file(get("--source"));
            Slang::ComPtr<ISlangBlob> diagnostics;
            report["stage"] = "load_module";
            auto module = session->loadModuleFromSourceString(
                "capability_probe",
                get("--source").c_str(),
                source.c_str(),
                diagnostics.writeRef()
            );
            diagnose(diagnostics);
            if (!module)
                throw std::runtime_error("Cannot load module");
            Slang::ComPtr<slang::IEntryPoint> entry;
            report["stage"] = "find_entry";
            check(module->findEntryPointByName(get("--entry", "main").c_str(), entry.writeRef()), nullptr);
            slang::IComponentType* components[] = {module, entry.get()};
            Slang::ComPtr<slang::IComponentType> composed;
            report["stage"] = "compose";
            auto result
                = session->createCompositeComponentType(components, 2, composed.writeRef(), diagnostics.writeRef());
            check(result, diagnostics);
            Slang::ComPtr<slang::IComponentType> linked;
            report["stage"] = "link";
            result = composed->link(linked.writeRef(), diagnostics.writeRef());
            check(result, diagnostics);
            Slang::ComPtr<ISlangBlob> code;
            report["stage"] = "codegen";
            result = get("--whole-program") == "1"
                ? linked->getTargetCode(0, code.writeRef(), diagnostics.writeRef())
                : linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef());
            check(result, diagnostics);
            std::ofstream output(get("--output"), std::ios::binary);
            output.write(static_cast<const char*>(code->getBufferPointer()), std::streamsize(code->getBufferSize()));
            if (!output)
                throw std::runtime_error("Cannot write output");
            report["output_bytes"] = std::to_string(code->getBufferSize());
        }
        report["status"] = "ok";
        report["stage"] = "complete";
        exit_code = 0;
    } catch (const std::exception& error) {
        report["error"] = error.what();
    }
#ifdef _WIN32
    for (const wchar_t* name : {L"dxcompiler.dll", L"dxil.dll", L"nvrtc64_120_0.dll", L"nvrtc64_130_0.dll"}) {
        if (auto module = GetModuleHandleW(name)) {
            wchar_t path[32768];
            auto count = GetModuleFileNameW(module, path, 32768);
            report["loaded:" + std::filesystem::path(name).string()]
                = std::filesystem::path(std::wstring(path, count)).string();
            if (auto version = GetProcAddress(module, "nvrtcVersion")) {
                int major = 0, minor = 0;
                reinterpret_cast<int (*)(int*, int*)>(version)(&major, &minor);
                report["nvrtc_version"] = std::to_string(major) + "." + std::to_string(minor);
            }
        }
    }
#endif
    std::cout << "{";
    bool first = true;
    for (const auto& [key, value] : report) {
        if (!first)
            std::cout << ',';
        std::cout << json_string(key) << ':' << json_string(value);
        first = false;
    }
    std::cout << "}\n";
    return exit_code;
}
