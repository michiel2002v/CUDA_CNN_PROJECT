cmake_minimum_required(VERSION 3.21)
project(CUDA_CNN_PROJECT CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDA_CNN_PROJECT main.cu)

set_target_properties(CUDA_CNN_PROJECT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
