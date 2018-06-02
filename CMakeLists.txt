cmake_minimum_required(VERSION 2.8.12)
project(fastlda)
add_subdirectory(pybind11)

include_directories(./src/)
pybind11_add_module(fastlda SHARED src/cpp/bindings.cpp src/cpp/lda.cpp
                            src/cpp/lda_inference.cpp
                            src/cpp/alias_table.cpp src/cpp/cgs_lda.cpp
                            src/cpp/lightlda.cpp)
