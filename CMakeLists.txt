cmake_minimum_required(VERSION 3.5)

project(locusta)

# The project version number.
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   1   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   0   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)


# Options
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_DOCS "Enable documentation generation" OFF)
option(ENABLE_TESTS "Enable testing" ON)

# Conditional CUDA support
if(ENABLE_CUDA)
  find_package(CUDAToolkit)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    set(CMAKE_CUDA_ARCHITECTURES "native")
  else()
    message(FATAL_ERROR "CUDA Toolkit not found, but ENABLE_CUDA is ON.")
  endif()
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_VERBOSE_MAKEFILE ON)

# C/C++ Flags
#set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb -g -pg -D_DEBUG -fdump-tree-all")
#set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -pg -fdump-tree-all")
#set(CMAKE_C_FLAGS_RELEASE "-O3")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -ffast-math -march=native -mtune=native -fPIC")

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

option(BUILD_SHARED_LIBS
  "On to build shared libraries, off for static libraries."
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

set(locusta_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  )


# include_directories(${locusta_INCLUDE_DIRS})


add_subdirectory (src)

if (ENABLE_TESTS)
  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  enable_testing()

  add_subdirectory(examples)
endif()


# Documentation
if(ENABLE_DOCS)
  find_package(Doxygen REQUIRED)

    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxy_main_page "${CMAKE_CURRENT_SOURCE_DIR}/Mainpage.dox")
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

