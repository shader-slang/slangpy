# Configure sanitizer instrumentation for SlangPy-owned native targets.

if(NOT SGL_ENABLE_ASAN AND NOT SGL_ENABLE_UBSAN)
    function(sgl_enable_sanitizers target)
    endfunction()
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "SlangPy sanitizers are only supported with Clang")
endif()

set(SGL_SANITIZERS)
if(SGL_ENABLE_ASAN)
    list(APPEND SGL_SANITIZERS address)
endif()
if(SGL_ENABLE_UBSAN)
    list(APPEND SGL_SANITIZERS undefined)
endif()
list(JOIN SGL_SANITIZERS "," SGL_SANITIZER_FLAGS)

add_library(sgl_sanitizers INTERFACE)
target_compile_options(sgl_sanitizers INTERFACE
    -fsanitize=${SGL_SANITIZER_FLAGS}
    -fno-omit-frame-pointer
)
target_link_options(sgl_sanitizers INTERFACE
    -fsanitize=${SGL_SANITIZER_FLAGS}
)

if(SGL_ENABLE_UBSAN)
    target_compile_options(sgl_sanitizers INTERFACE
        -fno-sanitize-recover=undefined
        -fsanitize-ignorelist=${CMAKE_SOURCE_DIR}/tools/ubsan-ignorelist.txt
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # Clang's Windows vptr runtime is built against the static release CRT,
        # while SlangPy and its vcpkg dependencies use the dynamic CRT. Keep the
        # remaining UBSan checks without introducing an incompatible runtime.
        # Clang autolinks both UBSan libraries for the umbrella `undefined`
        # group even after vptr is disabled. Ignore only the incompatible C++
        # runtime default library and explicitly retain the core runtime.
        target_compile_options(sgl_sanitizers INTERFACE
            -fno-sanitize=vptr
        )
        target_link_options(sgl_sanitizers INTERFACE
            -fno-sanitize=vptr
        )

        if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
            set(SGL_UBSAN_RUNTIME_ARCH "x86_64")
        elseif(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
            set(SGL_UBSAN_RUNTIME_ARCH "aarch64")
        else()
            message(FATAL_ERROR
                "Unsupported Windows architecture for UBSan: ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}"
            )
        endif()
        set(SGL_UBSAN_RUNTIME_NAME
            "clang_rt.ubsan_standalone-${SGL_UBSAN_RUNTIME_ARCH}.lib"
        )
        target_link_options(sgl_sanitizers INTERFACE
            -Xlinker /NODEFAULTLIB:clang_rt.ubsan_standalone_cxx-${SGL_UBSAN_RUNTIME_ARCH}.lib
        )
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=${SGL_UBSAN_RUNTIME_NAME}
            OUTPUT_VARIABLE SGL_UBSAN_RUNTIME
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT EXISTS "${SGL_UBSAN_RUNTIME}")
            message(FATAL_ERROR "Could not locate the Windows UBSan runtime: ${SGL_UBSAN_RUNTIME_NAME}")
        endif()
        target_link_libraries(sgl_sanitizers INTERFACE "${SGL_UBSAN_RUNTIME}")
    endif()
endif()

if(SGL_ENABLE_ASAN AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 22)
        message(FATAL_ERROR
            "Windows ASan requires Clang 22 or newer because older releases "
            "mis-instrument C++ exception catch parameters"
        )
    endif()

    # The debug CRT conflicts with ASan. SlangPy's vcpkg triplet uses the
    # dynamic release CRT, so keep all targets on that compatible runtime.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

    # Current Clang releases use the dynamic ASan runtime for Windows executables
    # and DLLs, including DLLs loaded by an unsanitized host. Prebuilt vcpkg
    # libraries have MSVC STL annotations disabled, so all linked objects must
    # use the same setting to satisfy the linker's /failifmismatch checks.
    target_compile_definitions(sgl_sanitizers INTERFACE
        _DISABLE_STRING_ANNOTATION
        _DISABLE_VECTOR_ANNOTATION
    )

    # Windows exception unwinding is still incompatible with ASan's general
    # stack-object instrumentation (google/sanitizers#749). Disabling stack
    # instrumentation retains heap and global checks and allows exceptions to
    # cross between the extension, the stock CPython DLL, and the test host.
    target_compile_options(sgl_sanitizers INTERFACE
        -mllvm -asan-stack=0
    )
endif()

function(sgl_enable_sanitizers target)
    target_link_libraries(${target} PRIVATE sgl_sanitizers)
endfunction()

message(STATUS "SlangPy sanitizers: ${SGL_SANITIZER_FLAGS}")
