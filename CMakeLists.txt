cmake_minimum_required(VERSION 2.8.12)
project(fastlda)
add_subdirectory(pybind11)

include_directories(.)
pybind11_add_module(fastlda SHARED
                            cpp/bindings.cpp cpp/lda.cpp cpp/lda_inference.cpp
                            cpp/alias_table.cpp cpp/cgs_lda.cpp
                            cpp/lightlda.cpp)
