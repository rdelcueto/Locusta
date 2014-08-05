#include <iostream>

#include "../src/prngenerator/prngenerator_gpu.h"
#include "../src/population_set_gpu.h"

#include "../src/evaluator/evaluator_gpu.h"
#include "../src/benchmarks/benchmarks_gpu.h"
#include "../src/ga/ga_solver_gpu.h"


//const uint THREADS = omp_get_num_threads();
const uint64_t SEED = 1;
const size_t ISLES = 64;
const size_t AGENTS = 128;
const size_t DIM = 32;
const bool MINIMIZE = false;
const size_t GENERATIONS = 1e2;
//const size_t TEST_ELEMENTS = 1e1;

int main(int argc, char *argv[])
{
  using namespace locusta;

  float upper_bounds[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  float lower_bounds[] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, \
                          -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, \
                          -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, \
                          -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, \
                          -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

  __setup_cuda();

  // Creating Pseudo Random Number Generator
  prngenerator_gpu<float> * generator = new prngenerator_gpu<float>(ISLES * AGENTS);
  //curandState * const dev_engines = generator->get_device_generator_states();

  generator->_initialize_engines(SEED);

  population_set_gpu<float> * population;
  population = new population_set_gpu<float>(ISLES,
                                                      AGENTS,
                                                      DIM,
                                                      upper_bounds,
                                                      lower_bounds);

  population->_initialize();

  // Creating Evaluator
  evaluator_gpu<float> * evaluator;
  evaluator = new evaluator_gpu<float>(true,
                                       evaluator_gpu<float>::IGNORE_BOUNDS,
                                       0,
                                       benchmark_gpu_func_1<float>);

  // Creating GA Solver
  ga_solver_gpu<float> * solver = new ga_solver_gpu<float>(population,
                                                           evaluator,
                                                           generator);

  // GA Solver Setup
  solver->_setup_operators(ga_operators_gpu<float>::tournament_select,
                           ga_operators_gpu<float>::whole_crossover,
                           ga_operators_gpu<float>::migration_ring);

  solver->_set_migration_config(10, // Migration step
                                1, // Migration size,
                                4
                                );

  solver->_set_selection_config(8, // Selection size,
                                0.1 // Selection stochastic bias
                                );

  solver->_set_breeding_config(0.8, // Crossover rate
                               0.1  // Mutation rate
                               );

  solver->_set_range_extension(0.1);

  solver->_initialize();

  solver->_print_solver_config();

  solver->_initialize_population();

// TODO: Target Generation
   for(size_t g = 0; g < GENERATIONS; ++g)
    {
#ifdef _DEBUG
      std::cout << "Generation: " << g << std::endl;
#else
      std::cout << "Generation: " << g << "\r";
#endif
      solver->_advance_generation();
      //solver->_print_solver_elite();
    }

   solver->_print_solver_solution();

   return 0;
}
