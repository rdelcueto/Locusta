include_directories(${CMAKE_SOURCE_DIR}/src/)

set(CUFILES
  benchmarks/kernel/benchmarks_gpu.cu
  )

#source_group("CUDA Source Files" FILES ${CUFILES}

LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

# Unit Tests
cuda_add_executable(runUnitTests
  locustaUnitTests.cpp
  ${CUFILES}
)

# Linking for Google Test.
target_link_libraries(runUnitTests ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# Locusta & Cuda linking.
target_link_libraries(runUnitTests locusta-cuda ${CUDA_curand_LIBRARY})

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
add_test(NAME test-population COMMAND runUnitTests)
