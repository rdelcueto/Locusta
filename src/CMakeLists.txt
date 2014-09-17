set(HFILES
  prngenerator/prngenerator.h
  population/population_set.h
  evaluator/evaluator.h
  solvers/evolutionary_solver.h
  solvers/ga/ga_solver.h
  solvers/ga/ga_operators.h
  )

set(CUFILES
  prngenerator/kernel/prngenerator_gpu.cu
  population/kernel/population_set_gpu.cu
  solvers/ga/kernel/ga_operators_gpu.cu
  )

source_group("CUDA Source Files" FILES ${CUFILES})

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
cuda_add_library(locusta-cuda
  ${CUFILES} 
)

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
