cmake_minimum_required(VERSION 3.5)

add_library(NeuralNetwork neuralNetwork.c)

target_link_libraries(NeuralNetwork Matrix ActivationFunctions LayerCreation)