find_package(Sphinx)

if(NOT SPHINX_FOUND)
    message("sphinx needs to be installed to generate documentation")
    return()
endif()

set(SPHINX_INPUT_DIR  "${CMAKE_CURRENT_SOURCE_DIR}")
set(SPHINX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

add_custom_target(doc
    ${Python_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/docs.py" build
        --api-mode runtime
        --builder html
        --output-dir "${SPHINX_OUTPUT_DIR}"
    COMMENT "Building HTML documentation with Sphinx"
    USES_TERMINAL
)

if(TARGET slangpy_ext)
    add_dependencies(doc slangpy_ext)
endif()
