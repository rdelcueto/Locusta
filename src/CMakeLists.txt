set(HPPFILES
  prngenerator/prngenerator.hpp
  prngenerator/prngenerator_cpu.hpp
  prngenerator/prngenerator_cuda.hpp
  population/population_set.hpp
  population/population_set_cpu.hpp
  population/population_set_cuda.hpp
  evaluator/evaluator.hpp
  evaluator/evaluator_cpu.hpp
  evaluator/evaluator_cuda.hpp
  solvers/evolutionary_solver.hpp
  solvers/evolutionary_solver_cpu.hpp
  solvers/evolutionary_solver_cuda.hpp
  solvers/pso/pso_solver_cpu.hpp
  solvers/pso/pso_operators/pso_operators.hpp
  solvers/pso/pso_solver_cuda.hpp
  solvers/pso/pso_operators/pso_operators_cuda.hpp
  solvers/ga/ga_solver_cpu.hpp
  solvers/ga/ga_operators/ga_operators.hpp
  solvers/ga/ga_solver_cuda.hpp
  solvers/ga/ga_operators/ga_operators_cuda.hpp
  solvers/ga/de_solver_cpu.hpp
  solvers/ga/de_operators/de_operators.hpp
  solvers/ga/de_solver_cuda.hpp
  solvers/ga/de_operators/de_operators_cuda.hpp

  )


set(CUFILES
  prngenerator/kernel/prngenerator_cuda.cu
  population/kernel/population_set_cuda.cu
  solvers/kernel/evolutionary_solver_cuda.cu
  solvers/pso/kernel/pso_solver_cuda.cu
  solvers/pso/pso_operators/kernel/pso_std_operators_cuda.cu
  solvers/ga/kernel/ga_solver_cuda.cu
  solvers/ga/ga_operators/kernel/ga_std_operators_cuda.cu
  solvers/de/kernel/de_solver_cuda.cu
  solvers/de/de_operators/kernel/de_std_operators_cuda.cu
  )

source_group("Locusta Source Files" FILES ${CUFILES} {HPPFILES})

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
cuda_add_library(locusta-cuda
  ${CUFILES}
)

INSTALL(FILES "${CMAKE_BINARY_DIR}/liblocusta-cuda.so"
  DESTINATION lib)

install(TARGETS locusta-cuda
  DESTINATION lib
  EXPORT locusta-targets
  )

install(EXPORT locusta-targets
  DESTINATION lib
  )
