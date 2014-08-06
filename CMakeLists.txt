project(locusta_test)

set(locusta_VERSION_MAJOR 0)
set(locusta_VERSION_MINOR 1)
set(locusta_VERSION_PATCH 0)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
set(CMAKE_VERBOSE_MAKEFILE ON)

add_definitions(-std=gnu++0x)

# Flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wextra -g -pg -D_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O2")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CUDA
find_package(CUDA REQUIRED)
find_cuda_helper_libs(curand)

option(BUILD_SHARED_LIBS	
  "On to build shared libraries, off for static libraries."
  OFF
  )

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF
  )

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored"
  )

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored"
  )

set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  )

set(locusta_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  )

include_directories(${locusta_INCLUDE_DIRS})

set(LIB_TYPE STATIC)

add_subdirectory (src)

# Testing
################################
if (test)
  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  enable_testing()

  add_subdirectory(tests)
endif()
