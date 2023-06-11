cmake_minimum_required(VERSION 3.5)

project(neuralcpp LANGUAGES CXX)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(sources)