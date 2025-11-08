macro(ternary var boolean value1 value2)
    if(${boolean})
        set(${var} ${value1})
    else()
        set(${var} ${value2})
    endif()
endmacro()

# Function to extract #define values from C/C++ headers
function(extract_define_from_header header_file define_name output_var)
    file(STRINGS "${header_file}" define_line REGEX "^[ \t]*#[ \t]*define[ \t]+${define_name}")

    # Strip C++ style comments
    string(REGEX REPLACE "//.*$" "" define_line "${define_line}")
    # Strip C style comments (basic, doesn't handle multiline)
    string(REGEX REPLACE "/\\*.*\\*/" "" define_line "${define_line}")

    # Handle string defines: #define FOO "value"
    if(define_line MATCHES "\"([^\"]*)\"")
        set(${output_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    # Handle numeric/unquoted defines: #define FOO 123
    elseif(define_line MATCHES "#define[ \t]+${define_name}[ \t]+([^ \t\n]+)")
        set(${output_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
        set(${output_var} "" PARENT_SCOPE)
    endif()
endfunction()
