cmake_minimum_required(VERSION 3.10)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId} CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# set(ENV{CUDA_VISIBLE_DEVICES} "0")  # Set this to the index of the device you want to use

add_definitions(-DNDEBUG)
# Set the build type to Release
#set(CMAKE_BUILD_TYPE "Release")


# Add the external library subdirectory
add_subdirectory(./partysecond/cudasupportfunctions)
add_subdirectory(./partythird/cuda)
add_subdirectory(./scripts)

# Set header files (C++ files)
file(GLOB_RECURSE _HDRS "include/*.h*")
file(GLOB_RECURSE _HDRS_cuda "include/*.cuh")

# Set source files (C++ files)
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")

# Set CUDA source files
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

# Set include directories
include_directories(include)

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Create shared library
cuda_add_library(${ProjectId} SHARED ${CPP_SOURCES} ${CUDA_SOURCES}
        include/Vector2File.h)
target_include_directories(${ProjectId} PUBLIC include)

# You can link against other libraries if needed
target_link_libraries(${ProjectId} cudasupportfunctions)

# Install the library
install(TARGETS ${ProjectId} DESTINATION lib)
install(FILES ${_HDRS} DESTINATION include/${ProjectId})
install(FILES ${_HDRS_cuda} DESTINATION include/${ProjectId})

