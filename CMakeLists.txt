cmake_minimum_required(VERSION 2.8.12)
project(fastlda)

add_subdirectory(pybind11)
pybind11_add_module(fastlda cpp/bindings.cpp cpp/lda.cpp)
