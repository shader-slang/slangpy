// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// -------------------------------------------------------------------------------------------------
// Compiler macros
// -------------------------------------------------------------------------------------------------

#define SGL_COMPILER_MSVC 1
#define SGL_COMPILER_CLANG 2
#define SGL_COMPILER_GCC 3

/**
 * Determine the compiler in use.
 * http://sourceforge.net/p/predef/wiki/Compilers/
 */
#ifndef SGL_COMPILER
#if defined(_MSC_VER) && !defined(__clang__)
#define SGL_COMPILER SGL_COMPILER_MSVC
#elif defined(__clang__)
#define SGL_COMPILER SGL_COMPILER_CLANG
#elif defined(__GNUC__)
#define SGL_COMPILER SGL_COMPILER_GCC
#else
#error "Unsupported compiler"
#endif
#endif // SGL_COMPILER

#define SGL_MSVC (SGL_COMPILER == SGL_COMPILER_MSVC)
#define SGL_CLANG (SGL_COMPILER == SGL_COMPILER_CLANG)
#define SGL_GCC (SGL_COMPILER == SGL_COMPILER_GCC)

// -------------------------------------------------------------------------------------------------
// Architecture macros
// -------------------------------------------------------------------------------------------------

#define SGL_ARCH_X86_64 1
#define SGL_ARCH_ARM64 2

/**
 * Determine the target architecture in use.
 * http://sourceforge.net/p/predef/wiki/Architectures/
 */
#ifndef SGL_ARCH
#if defined(_M_X64) || defined(__x86_64__)
#define SGL_ARCH SGL_ARCH_X86_64
#elif defined(_M_ARM64) || defined(__aarch64__)
#define SGL_ARCH SGL_ARCH_ARM64
#else
#error "Unsupported target architecture"
#endif
#endif // SGL_ARCH

#define SGL_X86_64 (SGL_ARCH == SGL_ARCH_X86_64)
#define SGL_ARM64 (SGL_ARCH == SGL_ARCH_ARM64)

// -------------------------------------------------------------------------------------------------
// Platform macros
// -------------------------------------------------------------------------------------------------

#define SGL_PLATFORM_WINDOWS 1
#define SGL_PLATFORM_LINUX 2
#define SGL_PLATFORM_MACOS 3

/**
 * Determine the target platform in use.
 * http://sourceforge.net/p/predef/wiki/OperatingSystems/
 */
#ifndef SGL_PLATFORM
#if defined(_WIN64)
#define SGL_PLATFORM SGL_PLATFORM_WINDOWS
#elif defined(__linux__)
#define SGL_PLATFORM SGL_PLATFORM_LINUX
#elif defined(__APPLE__) && defined(__MACH__)
#define SGL_PLATFORM SGL_PLATFORM_MACOS
#else
#error "Unsupported target platform"
#endif
#endif // SGL_PLATFORM

#define SGL_WINDOWS (SGL_PLATFORM == SGL_PLATFORM_WINDOWS)
#define SGL_LINUX (SGL_PLATFORM == SGL_PLATFORM_LINUX)
#define SGL_MACOS (SGL_PLATFORM == SGL_PLATFORM_MACOS)

/**
 * Shared library (DLL) export and import.
 */
#if SGL_WINDOWS
#define SGL_API_EXPORT __declspec(dllexport)
#define SGL_API_IMPORT __declspec(dllimport)
#else
#define SGL_API_EXPORT __attribute__((visibility("default")))
#define SGL_API_IMPORT
#endif

#ifdef SGL_DLL
#define SGL_API SGL_API_EXPORT
#else // SGL_DLL
#define SGL_API SGL_API_IMPORT
#endif // SGL_DLL

// -------------------------------------------------------------------------------------------------
// Convenience macros
// -------------------------------------------------------------------------------------------------

#if SGL_MSVC
#define SGL_INLINE __forceinline
#elif SGL_CLANG | SGL_GCC
#define SGL_INLINE __attribute__((always_inline))
#endif

#define SGL_NON_COPYABLE(cls)                                                                                          \
    cls(const cls&) = delete;                                                                                          \
    cls& operator=(const cls&) = delete

#define SGL_NON_MOVABLE(cls)                                                                                           \
    cls(cls&&) = delete;                                                                                               \
    cls& operator=(cls&&) = delete

#define SGL_NON_COPYABLE_AND_MOVABLE(cls)                                                                              \
    SGL_NON_COPYABLE(cls);                                                                                             \
    SGL_NON_MOVABLE(cls)

// clang-format off
namespace sgl::detail { template<typename... Args> inline void unused(Args&&...) {} }
// clang-format on

#define SGL_UNUSED(...) ::sgl::detail::unused(__VA_ARGS__)

#define SGL_STRINGIZE(a) #a
#define SGL_TO_STRING(a) SGL_STRINGIZE(a)
#define SGL_CONCAT_STRINGS_(a, b) a##b
#define SGL_CONCAT_STRINGS(a, b) SGL_CONCAT_STRINGS_(a, b)

// -------------------------------------------------------------------------------------------------
// Compiler warnings
// -------------------------------------------------------------------------------------------------

// Example usage:
// SGL_DIAGNOSTIC_PUSH
// SGL_DISABLE_MSVC_WARNING(4455)
// SGL_DISABLE_CLANG_WARNING("-Wuser-defined-literals")
// SGL_DISABLE_GCC_WARNING("-Wliteral-suffix")
// inline float16_t operator""h(long double value) { return float16_t(static_cast<float>(value)); }
// SGL_DIAGNOSTIC_POP

#if SGL_MSVC
#define SGL_DIAGNOSTIC_PUSH __pragma(warning(push))
#define SGL_DIAGNOSTIC_POP __pragma(warning(pop))
#define SGL_DISABLE_WARNING(warning_) __pragma(warning(disable : warning_))
#define SGL_DISABLE_MSVC_WARNING(warning) SGL_DISABLE_WARNING(warning)
#define SGL_DISABLE_CLANG_WARNING(warning)
#define SGL_DISABLE_GCC_WARNING(warning)
#elif SGL_CLANG
#define SGL_DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#define SGL_DIAGNOSTIC_POP _Pragma("clang diagnostic pop")
#define SGL_DISABLE_WARNING(warning) _Pragma(SGL_STRINGIZE(clang diagnostic ignored warning))
#define SGL_DISABLE_MSVC_WARNING(warning)
#define SGL_DISABLE_CLANG_WARNING(warning) SGL_DISABLE_WARNING(warning)
#define SGL_DISABLE_GCC_WARNING(warning)
#elif SGL_GCC
#define SGL_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define SGL_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#define SGL_DISABLE_WARNING(warning) _Pragma(SGL_STRINGIZE(GCC diagnostic ignored warning))
#define SGL_DISABLE_MSVC_WARNING(warning)
#define SGL_DISABLE_CLANG_WARNING(warning)
#define SGL_DISABLE_GCC_WARNING(warning) SGL_DISABLE_WARNING(warning)
#endif

// -------------------------------------------------------------------------------------------------
// Enum flags
// -------------------------------------------------------------------------------------------------

// clang-format off
/// Implement logical operators on a class enum for making it usable as a flags enum.
#define SGL_ENUM_CLASS_OPERATORS(e_) \
    inline e_ operator& (e_ a, e_ b) { return static_cast<e_>(static_cast<int>(a)& static_cast<int>(b)); } \
    inline e_ operator| (e_ a, e_ b) { return static_cast<e_>(static_cast<int>(a)| static_cast<int>(b)); } \
    inline e_& operator|= (e_& a, e_ b) { a = a | b; return a; }; \
    inline e_& operator&= (e_& a, e_ b) { a = a & b; return a; }; \
    inline e_  operator~ (e_ a) { return static_cast<e_>(~static_cast<int>(a)); } \
    inline bool is_set(e_ val, e_ flag) { return (val & flag) != static_cast<e_>(0); } \
    inline void flip_bit(e_& val, e_ flag) { val = is_set(val, flag) ? (val & (~flag)) : (val | flag); }
// clang-format on
