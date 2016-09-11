project(locusta_test)

# The project version number.
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   1   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   0   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
set(CMAKE_VERBOSE_MAKEFILE ON)

# C/C++ Flags
add_definitions(-std=c++11)
set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb -g -pg -D_DEBUG -fdump-tree-all")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -pg -fdump-tree-all")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -ffast-math -march=native -mtune=native -fPIC")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g -pg -D_DEBUG -fdump-tree-all")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -pg -fdump-tree-all")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ffast-math -march=native -mtune=native -fPIC")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
 set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
 set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ENABLE_SIMD
option(ENABLE_SIMD
  "On to compile with SIMD instructions."
  ON
  )

if (ENABLE_SIMD)
  #set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fopt-info-vec-all")
  #set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fopt-info-vec-all")
else ()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-tree-vectorize -fno-tree-loop-vectorize")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-tree-vectorize -fno-tree-loop-vectorize")
endif()

# OpenMP SIMD
if (OPENMP_FOUND)
  if (ENABLE_SIMD)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp-simd")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-simd")
  else()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-openmp-simd")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-openmp-simd")
  endif()
endif()

# ENABLE Prefetched loops
option(ENABLE_PREFETCH
  "On to compile with loop prefetching."
  ON
  )

if (ENABLE_PREFETCH)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprefetch-loop-arrays")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprefetch-loop-arrays")
endif()

# CUDA
set(CUDA_HOST_COMPILER /usr/bin/g++-5)
find_package(CUDA REQUIRED)
find_cuda_helper_libs(curand)

# CUDA Arch Target

# PRE-FERMI (DEPRECATED)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)

# ALL POST-FERMI
LIST(APPEND CUDA_NVCC_FLAGS
  -Xcompiler -fPIC
  -Xptxas -v
  # -gencode=arch=compute_20,code=sm_20
  # -gencode=arch=compute_30,code=sm_30
  # -gencode=arch=compute_35,code=sm_35
  # -gencode=arch=compute_50,code=sm_50
  -gencode=arch=compute_52,code=sm_52
  #--ptxas-options=-v
  --use_fast_math
  --restrict
  -O3
  -pg
  -G
  )

LIST(APPEND CUDA_NVCC_FLAGS -std=c++11)
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
option(BUILD_TESTS
  "On to build test suite."
  ON
  )

if (BUILD_TESTS)
  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  enable_testing()

  add_subdirectory(tests)
endif()


# Documentation
option(BUILD_DOCS
  "On to build doxygen documentation."
  OFF
  )

if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)

    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)

endif()
