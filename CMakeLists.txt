cmake_minimum_required(VERSION 3.22)


set (CMAKE_CXX_FLAGS "--hipsycl-targets='omp;cuda:sm_60'")
set (CMAKE_CXX_FLAGS_RELEASE "-O2")
set (CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_COMPILER "syclcc-clang")


project(Accelerated-CA-SYCL VERSION 0.1)
add_executable(ca_main ca_main.cxx)

# Used for ycm completion
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
