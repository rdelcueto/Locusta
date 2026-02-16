
# Common Headers (CPU)
set(HPPFILES
  prngenerator/prngenerator.hpp
  prngenerator/prngenerator_cpu.hpp
  population/population_set.hpp
  population/population_set_cpu.hpp
  evaluator/evaluator.hpp
  evaluator/evaluator_cpu.hpp
  solvers/evolutionary_solver.hpp
  solvers/evolutionary_solver_cpu.hpp
  solvers/pso/pso_solver_cpu.hpp
  solvers/pso/pso_operators/pso_operators.hpp
  solvers/ga/ga_solver_cpu.hpp
  solvers/ga/ga_operators/ga_operators.hpp
  solvers/de/de_solver_cpu.hpp
  solvers/de/de_operators/de_operators.hpp
  )

# Add headers to a common source group for IDEs
source_group("Locusta CPU Files" FILES ${HPPFILES})

# Interface library for CPU-only usage (header-only)
add_library(locusta INTERFACE)
target_include_directories(locusta INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
# If you have cpp files for CPU, you would change INTERFACE to STATIC/SHARED and add sources here.

install(TARGETS locusta
  EXPORT locusta-targets
  DESTINATION lib
)

if(ENABLE_CUDA)
  set(CU_HPPFILES
    prngenerator/prngenerator_cuda.hpp
    population/population_set_cuda.hpp
    evaluator/evaluator_cuda.hpp
    solvers/evolutionary_solver_cuda.hpp
    solvers/pso/pso_solver_cuda.hpp
    solvers/pso/pso_operators/pso_operators_cuda.hpp
    solvers/ga/ga_solver_cuda.hpp
    solvers/ga/ga_operators/ga_operators_cuda.hpp
    solvers/de/de_solver_cuda.hpp
    solvers/de/de_operators/de_operators_cuda.hpp
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

  source_group("Locusta CUDA Files" FILES ${CUFILES} ${CU_HPPFILES})

  if (CUDA_VERBOSE_PTXAS)
    set(VERBOSE_PTXAS --ptxas-options=-v)
  endif (CUDA_VERBOSE_PTXAS)

  #set(CUDA_SEPARABLE_COMPILATION ON)
  #set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
  
  add_library(locusta-cuda STATIC ${CUFILES} ${CU_HPPFILES})

  # Link against common locusta interface
  target_link_libraries(locusta-cuda PUBLIC locusta)

  install(TARGETS locusta-cuda
    EXPORT locusta-targets
    DESTINATION lib
    )
endif()

install(EXPORT locusta-targets
  DESTINATION lib
  )

