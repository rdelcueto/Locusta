
# --- CPU Example ---
add_executable(locustaExamples-cpu locustaExamples_cpu.cpp)
target_compile_definitions(locustaExamples-cpu PRIVATE LOCUSTA_CPU_ONLY)
target_link_libraries(locustaExamples-cpu PRIVATE locusta ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

add_test(NAME test-cpu COMMAND locustaExamples-cpu)

# --- CUDA Example ---
if(ENABLE_CUDA)
  set(CUFILES benchmarks/kernel/benchmarks_cuda.cu)
  
  add_executable(locustaExamples-cuda locustaExamples_cuda.cpp ${CUFILES})
  target_compile_definitions(locustaExamples-cuda PRIVATE LOCUSTA_ENABLE_CUDA)
  target_link_libraries(locustaExamples-cuda PRIVATE locusta locusta-cuda ${CUDA_curand_LIBRARY} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

  add_test(NAME test-cuda COMMAND locustaExamples-cuda)
endif()


