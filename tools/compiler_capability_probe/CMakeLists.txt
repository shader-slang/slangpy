cmake_minimum_required(VERSION 3.20)
project(compiler_capability_probe LANGUAGES CXX)

set(SLANG_INCLUDE_DIR "" CACHE PATH "Headers for the Slang API under investigation")
if(NOT EXISTS "${SLANG_INCLUDE_DIR}/slang.h")
    message(FATAL_ERROR "Set SLANG_INCLUDE_DIR to a directory containing slang.h")
endif()

add_executable(compiler_capability_probe main.cpp)
target_compile_features(compiler_capability_probe PRIVATE cxx_std_17)
target_include_directories(compiler_capability_probe PRIVATE "${SLANG_INCLUDE_DIR}")
target_link_libraries(compiler_capability_probe PRIVATE ${CMAKE_DL_LIBS})
