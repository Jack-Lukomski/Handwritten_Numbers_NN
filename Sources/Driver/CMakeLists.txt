cmake_minimum_required(VERSION 3.5)

project(driver LANGUAGES C)

add_executable(driver main.c)

target_link_libraries(driver PRIVATE matrix)

set_target_properties(driver PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)
