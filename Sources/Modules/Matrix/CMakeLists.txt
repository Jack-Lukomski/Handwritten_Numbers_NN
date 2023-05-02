cmake_minimum_required(VERSION 3.5)

project(matrix LANGUAGES C)

add_library(matrix matrix.c)

set_target_properties(matrix PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)
