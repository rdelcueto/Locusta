include_directories(${CMAKE_SOURCE_DIR}/src/)

set(CUFILES
  benchmarks/kernel/benchmarks_gpu.cu
  )

#source_group("CUDA Source Files" FILES ${CUFILES})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM11 -gencode=arch=compute_11,code=sm_11 -gencode=arch=compute_11,code=compute_11)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)

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
