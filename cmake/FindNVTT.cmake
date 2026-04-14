# FindNVTT.cmake - Find NVIDIA Texture Tools 3
#
# This module finds NVTT3 headers, import library, and shared library.
#
# Hints:
#   SGL_NVTT_PREFIX  - Root directory of the NVTT installation
#
# The module also searches in ${CMAKE_SOURCE_DIR}/external/nvtt by default.
#
# Result variables:
#   NVTT_FOUND         - True if NVTT was found
#   NVTT_INCLUDE_DIRS  - Include directories
#   NVTT_LIBRARIES     - Import library path
#   NVTT_DLL           - Shared library (DLL) path
#   NVTT_VERSION       - Version string (e.g. "3.2.5")
#
# Imported targets:
#   NVTT::NVTT         - Shared imported target

if(NVTT_FOUND)
    return()
endif()

set(_nvtt_search_paths "")
if(SGL_NVTT_PREFIX)
    list(APPEND _nvtt_search_paths "${SGL_NVTT_PREFIX}")
endif()
list(APPEND _nvtt_search_paths "${CMAKE_SOURCE_DIR}/external/nvtt")

# --- Header ---
find_path(NVTT_INCLUDE_DIR
    NAMES nvtt/nvtt_wrapper.h
    PATHS ${_nvtt_search_paths}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

# --- Import library ---
if(WIN32)
    # The lib is versioned (e.g. nvtt30205.lib). Search with a glob pattern.
    set(_nvtt_lib_suffixes lib/x64-v142 lib)
    find_library(NVTT_LIBRARY
        NAMES nvtt30205
        PATHS ${_nvtt_search_paths}
        PATH_SUFFIXES ${_nvtt_lib_suffixes}
        NO_DEFAULT_PATH
    )
    # Fallback: try any nvtt*.lib via glob if the exact name wasn't found.
    if(NOT NVTT_LIBRARY)
        foreach(_search_path ${_nvtt_search_paths})
            foreach(_suffix ${_nvtt_lib_suffixes})
                file(GLOB _nvtt_libs "${_search_path}/${_suffix}/nvtt*.lib")
                if(_nvtt_libs)
                    list(GET _nvtt_libs 0 NVTT_LIBRARY)
                    break()
                endif()
            endforeach()
            if(NVTT_LIBRARY)
                break()
            endif()
        endforeach()
    endif()
elseif(APPLE)
    find_library(NVTT_LIBRARY
        NAMES nvtt
        PATHS ${_nvtt_search_paths}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH
    )
else()
    find_library(NVTT_LIBRARY
        NAMES nvtt
        PATHS ${_nvtt_search_paths}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
    )
endif()

# --- Shared library (DLL / .so / .dylib) ---
if(WIN32)
    # DLL may be in root, bin/, or lib/
    set(NVTT_DLL "")
    foreach(_search_path ${_nvtt_search_paths})
        foreach(_suffix "" bin lib)
            if(_suffix)
                file(GLOB _nvtt_dlls "${_search_path}/${_suffix}/nvtt*.dll")
            else()
                file(GLOB _nvtt_dlls "${_search_path}/nvtt*.dll")
            endif()
            if(_nvtt_dlls)
                list(GET _nvtt_dlls 0 NVTT_DLL)
                break()
            endif()
        endforeach()
        if(NVTT_DLL)
            break()
        endif()
    endforeach()
endif()

# --- Version detection ---
set(NVTT_VERSION "")
if(NVTT_INCLUDE_DIR)
    # nvtt.h defines: #define NVTT_VERSION 30205  (= 10000*fork + 100*major + minor)
    set(_nvtt_version_header "${NVTT_INCLUDE_DIR}/nvtt/nvtt.h")
    if(EXISTS "${_nvtt_version_header}")
        file(STRINGS "${_nvtt_version_header}" _nvtt_version_line REGEX "^#define NVTT_VERSION [0-9]+")
        if(_nvtt_version_line)
            string(REGEX REPLACE "^#define NVTT_VERSION ([0-9]+)" "\\1" _nvtt_version_num "${_nvtt_version_line}")
            math(EXPR _nvtt_major "${_nvtt_version_num} / 10000")
            math(EXPR _nvtt_minor "(${_nvtt_version_num} % 10000) / 100")
            math(EXPR _nvtt_patch "${_nvtt_version_num} % 100")
            set(NVTT_VERSION "${_nvtt_major}.${_nvtt_minor}.${_nvtt_patch}")
        endif()
    endif()
endif()

# --- Standard find_package handling ---
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTT
    REQUIRED_VARS NVTT_INCLUDE_DIR NVTT_LIBRARY
    VERSION_VAR NVTT_VERSION
)

if(NVTT_FOUND)
    set(NVTT_INCLUDE_DIRS "${NVTT_INCLUDE_DIR}")
    set(NVTT_LIBRARIES "${NVTT_LIBRARY}")
    set(NVTT_DLL "${NVTT_DLL}" CACHE FILEPATH "Path to the NVTT shared library (DLL)" FORCE)

    if(NOT TARGET NVTT::NVTT)
        add_library(NVTT::NVTT SHARED IMPORTED)
        set_target_properties(NVTT::NVTT PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NVTT_INCLUDE_DIR}"
            IMPORTED_IMPLIB "${NVTT_LIBRARY}"
        )
        if(WIN32 AND NVTT_DLL)
            set_target_properties(NVTT::NVTT PROPERTIES
                IMPORTED_LOCATION "${NVTT_DLL}"
            )
        elseif(NOT WIN32 AND NVTT_LIBRARY)
            set_target_properties(NVTT::NVTT PROPERTIES
                IMPORTED_LOCATION "${NVTT_LIBRARY}"
            )
        endif()
    endif()
endif()

mark_as_advanced(NVTT_INCLUDE_DIR NVTT_LIBRARY NVTT_DLL)
