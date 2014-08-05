set(CUFILES
  population/kernel/population_set_gpu.cu
  solvers/ga/kernel/ga_operators_gpu.cu
  prngenerator/kernel/prngenerator_gpu.cu
  )

source_group("CUDA Source Files" FILES ${CUFILES})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM11 -gencode=arch=compute_11,code=sm_11 -gencode=arch=compute_11,code=compute_11)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

cuda_add_library(locusta-cuda
  ${CUFILES} 
  OPTIONS ${GENCODE_SM11})

INSTALL(FILES ${HFILES}
  DESTINATION include)
INSTALL(FILES "${CMAKE_BINARY_DIR}/liblocusta-cuda.so" 
  DESTINATION lib)

install(TARGETS locusta-cuda
  DESTINATION lib
  EXPORT locusta-targets
  )

install(EXPORT locusta-targets
  DESTINATION lib
  )
