#include <iostream>

#include "../benchmarks/benchmarks_cpu.h"

#include "evaluator/evaluator_cpu.h"

#include "population/population_set_cpu.h"
#include "prngenerator/prngenerator_cpu.h"
#include "solvers/ga/ga_solver_cpu.h"

//const size_t THREADS = omp_get_num_threads();
const uint64_t SEED = 1;
const size_t ISLES = 64;
const size_t AGENTS = 128;
const size_t DIM = 32;
const bool MINIMIZE = false;
const size_t GENERATIONS = 1e2;

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

  // Creating Population
  population_set_cpu<float> * population;
  population = new population_set_cpu<float>(ISLES,
                                                      AGENTS,
                                                      DIM,
                                                      upper_bounds,
                                                      lower_bounds);

  // Creating Evaluator
  evaluator_cpu<float> * evaluator;
  evaluator = new evaluator_cpu<float>(true,
                                       evaluator_cpu<float>::IGNORE_BOUNDS,
                                       0,
                                       benchmark_cpu_func_1);

  // Creating Pseudo Random Number Generator
  prngenerator_cpu<float> * generator = new prngenerator_cpu<float>(8);
  generator->_initialize_engines(SEED);

  // Creating GA Solver
  ga_solver_cpu<float> * solver = new ga_solver_cpu<float>(population,
                                                           evaluator,
                                                           generator);

  // GA Solver Setup
  solver->_setup_operators(ga_operators_cpu<float>::tournament_select,
                           ga_operators_cpu<float>::whole_crossover,
                           ga_operators_cpu<float>::migration_ring);

  solver->_set_migration_config(10, // Migration step
                                1, // Migration size,
                                4 // Migration selection size,
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
  std::cout << std::endl;

  return 0;
}
