cmake_minimum_required(VERSION 3.10)

SET(ProjectId "Scripts")
project(${ProjectId})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

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

file( GLOB APP_SOURCES "./*.cpp" )
message("====================================================================}")
message("APP_SOURCES: ${APP_SOURCES}")
message("====================================================================}")

get_filename_component(PARENT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
get_filename_component(PARENT_FOLDER_NAME "${PARENT_DIRECTORY}" NAME)

foreach( testsourcefile ${APP_SOURCES} )
    # Cut off the file extension and directory path
    get_filename_component( testname ${testsourcefile} NAME_WE )
    add_executable( ${testname} ${testsourcefile})
    
    # Make sure YourLib is linked to each app
    # You can link against other libraries if needed
    target_link_libraries(${testname} ${PARENT_FOLDER_NAME})
    target_link_libraries(${testname} cudasupportfunctions)

    target_link_libraries(${testname} CUDA::cudart)
    target_link_libraries(${testname} CUDA::cublas)
    target_link_libraries(${testname} CUDA::cufft)

    link_directories(${CMAKE_SOURCE_DIR}/install/lib)
    link_directories(${CMAKE_SOURCE_DIR}/build)
    
    # Install the ProjectA executable
    install(TARGETS ${testname} DESTINATION bin)
    message("testname: ${testname}")
endforeach( testsourcefile ${APP_SOURCES} )



