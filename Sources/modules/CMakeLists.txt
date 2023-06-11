cmake_minimum_required(VERSION 3.5)

add_library(layer Layer.cpp)
add_library(nn NeuralNetwork.cpp)
add_library(activationfunctions ActivationFunctions.cpp)

target_link_libraries(nn layer activationfunctions)