cmake_minimum_required(VERSION 3.5)

add_executable(driver main.c)

target_link_libraries(driver NeuralNetwork)
