include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CUFILES
  benchmarks/kernel/benchmarks_cuda.cu
  )

# Unit Tests
add_executable(runUnitTests
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
