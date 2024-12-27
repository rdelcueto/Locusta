include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CUFILES
  benchmarks/kernel/benchmarks_cuda.cu
  )

# Examples
add_executable(locustaExamples
  locustaExamples.cpp
  ${CUFILES}
)

# Linking for Google Test.
target_link_libraries(locustaExamples ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# Locusta & Cuda linking.
target_link_libraries(locustaExamples locusta-cuda ${CUDA_curand_LIBRARY})

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable locustaExamples to see those specific tests.
add_test(NAME test-population COMMAND locustaExamples)
