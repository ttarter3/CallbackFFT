
cmake_minimum_required(VERSION 3.17)

# Get the name of the parent directory
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Set the project name
project(${PROJECT_NAME})

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found at: ${CUDAToolkit_LIBRARY_DIR}")
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION_STRING}")
else()
    message(FATAL_ERROR "CUDA toolkit not found.")
endif()
