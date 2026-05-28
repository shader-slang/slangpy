find_package(Sphinx)

if(NOT SPHINX_FOUND)
    message("sphinx needs to be installed to generate documentation")
    return()
endif()

set(SPHINX_INPUT_DIR  "${CMAKE_CURRENT_SOURCE_DIR}")
set(SPHINX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

add_custom_target(doc
    ${SPHINX_EXECUTABLE} -b html "${SPHINX_INPUT_DIR}" "${SPHINX_OUTPUT_DIR}"
    COMMENT "Building HTML documentation with Sphinx"
    USES_TERMINAL
)
