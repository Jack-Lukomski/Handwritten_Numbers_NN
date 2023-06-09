cmake_minimum_required(VERSION 3.5)

add_executable(neuralcpp main.cpp)

target_link_libraries(neuralcpp layer)
target_link_libraries(neuralcpp ${ARMADILLO_LIBRARIES})