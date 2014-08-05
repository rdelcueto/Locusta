add_executable(ga_cpu_test
  ga_cpu_test.cpp)
cuda_add_executable(ga_gpu_test
  ga_gpu_test.cpp)
target_link_libraries(ga_gpu_test locusta-cuda locusta-benchmarks ${CUDA_curand_LIBRARY})
