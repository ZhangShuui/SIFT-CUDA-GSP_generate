cmake_minimum_required(VERSION 3.20)
project(cudanewts CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cudanewts main.cu)


set_target_properties(cudanewts PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

