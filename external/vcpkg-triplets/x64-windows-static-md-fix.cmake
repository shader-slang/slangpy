# Allow user to explicitly specify a toolset version via -DSGL_MSVC_TOOLSET_VERSION=...
# If not specified, it falls back to the VCToolsVersion from the developer environment.
if(DEFINED SGL_MSVC_TOOLSET_VERSION)
    set(toolset_to_use ${SGL_MSVC_TOOLSET_VERSION})
    message(STATUS "Triplet: Using user-specified toolset version '${toolset_to_use}'.")
elseif(DEFINED ENV{VCToolsVersion})
    set(toolset_to_use "$ENV{VCToolsVersion}")
    message(STATUS "Triplet: Using toolset version from environment: '${toolset_to_use}'.")
endif()

if(DEFINED toolset_to_use)
    # Set the detailed toolset version. This forces vcpkg to use the specific toolset.
    set(VCPKG_PLATFORM_TOOLSET_VERSION "${toolset_to_use}")

    # Also set the major toolset version, as vcpkg may require it to be present.
    # For VS 2022, all versions (14.30-14.49) use v143 toolset
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _match "${toolset_to_use}")
    if(_match)
        # Map version to correct Visual Studio toolset
        # 14.0-14.9 -> v140 (VS 2015)
        # 14.1X -> v141 (VS 2017)
        # 14.2X -> v142 (VS 2019)
        # 14.3X-14.4X -> v143 (VS 2022)
        if(CMAKE_MATCH_1 EQUAL 14)
            if(CMAKE_MATCH_2 GREATER_EQUAL 0 AND CMAKE_MATCH_2 LESS 10)
                set(derived_toolset "v140")
            elseif(CMAKE_MATCH_2 GREATER_EQUAL 10 AND CMAKE_MATCH_2 LESS 20)
                set(derived_toolset "v141")
            elseif(CMAKE_MATCH_2 GREATER_EQUAL 20 AND CMAKE_MATCH_2 LESS 30)
                set(derived_toolset "v142")
            else()
                # v143 for 14.30+ (VS 2022 and future versions)
                # Covers 14.3X, 14.4X, and beyond
                set(derived_toolset "v143")
            endif()
        else()
            # Fallback for unknown versions
            set(derived_toolset "v${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        endif()
        set(VCPKG_PLATFORM_TOOLSET "${derived_toolset}")
    endif()
endif()

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
# _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR avoids the use of constexpr mutex constructor
# in vcpkg packages, which can lead to binary incompatibility issues.
set(VCPKG_C_FLAGS "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
set(VCPKG_CXX_FLAGS "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
