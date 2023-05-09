cmake_minimum_required(VERSION 3.5)

add_library(LayerCreation layerCreation.c)
target_link_libraries(LayerCreation Matrix)