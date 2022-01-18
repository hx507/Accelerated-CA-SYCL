cmake_minimum_required(VERSION 3.22)


set (CMAKE_CXX_FLAGS "--hipsycl-targets='omp;cuda:sm_60'")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
set (CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_COMPILER "syclcc-clang")


# Used for ycm completion
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
if(DEFINED HIP_SYCL_INCLUDE_DIR)
    MESSAGE(STATUS "Adding SYCL include dir for ycm: $${HIP_SYCL_INCLUDE_DIR}")
    string(APPEND CMAKE_CXX_FLAGS " -I'${HIP_SYCL_INCLUDE_DIR}'")
endif(DEFINED HIP_SYCL_INCLUDE_DIR)


project(Accelerated-CA-SYCL VERSION 0.1)
add_executable(ca_main ca_main.cxx)
target_include_directories(ca_main PRIVATE "include")
