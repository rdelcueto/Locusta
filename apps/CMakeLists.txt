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

add_executable(ga_cpu_test
  tests/ga_cpu_test.cpp)

cuda_add_executable(ga_gpu_test
  tests/ga_gpu_test.cpp
  ${CUFILES}
)

target_link_libraries(ga_gpu_test locusta-cuda ${CUDA_curand_LIBRARY})
