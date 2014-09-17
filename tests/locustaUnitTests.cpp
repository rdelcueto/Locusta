#include "gtest/gtest.h"

#include <iostream>
#include <algorithm>
#include <time.h>

#include "benchmarks/benchmarks_cpu.hpp"
#include "benchmarks/benchmarks_gpu.hpp"

#include "prngenerator/prngenerator_cpu.hpp"
#include "prngenerator/prngenerator_gpu.hpp"

#include "population/population_set_cpu.hpp"
#include "population/population_set_gpu.hpp"

#include "evaluator/evaluator_cpu.hpp"
#include "evaluator/evaluator_gpu.hpp"

#include "solvers/ga/ga_solver_cpu.hpp"
#include "solvers/ga/ga_solver_gpu.hpp"

#include "cuda_runtime.h"

using namespace locusta;

class LocustaTest : public testing::Test {
protected:

  void setupCPU() {
    // Define Pseudo random generator
    cpu_generator = new prngenerator_cpu<float>(ISLES * AGENTS);
    cpu_generator->_initialize_engines(SEED);

    // Define Population
    cpu_population = new population_set_cpu<float>(ISLES,
                                                   AGENTS,
                                                   DIMENSIONS,
                                                   upper_bounds,
                                                   lower_bounds);

    // Initialize Population
    const uint32_t pop_size = DIMENSIONS * AGENTS * ISLES;
    float * const pop_data = cpu_population->_get_transformed_data_array();

    cpu_generator->_generate(pop_size, pop_data);
    cpu_population->_initialize();

    // Define Genome evaluator
    cpu_evaluator = new evaluator_cpu<float>(true,
                                             evaluator_cpu<float>::IGNORE_BOUNDS,
                                             0,
                                             benchmark_cpu_func_1<float>);
  }

  void setupGPU() {
    __setup_cuda();

    // Test data allocation
    gpu_fitness_test_bridge = new float[fitness_array_size];
    gpu_data_test_bridge = new float[data_array_size];

    // Invalidating initialized array values
    std::fill(gpu_fitness_test_bridge,
              gpu_fitness_test_bridge + fitness_array_size,
              std::numeric_limits<float>::quiet_NaN());

    std::fill(gpu_data_test_bridge,
              gpu_data_test_bridge + data_array_size,
              std::numeric_limits<float>::quiet_NaN());

    // Define Pseudo random generator
    gpu_generator = new prngenerator_gpu<float>(ISLES * AGENTS);
    // Define Population
    gpu_population = new population_set_gpu<float>(ISLES,
                                                   AGENTS,
                                                   DIMENSIONS,
                                                   upper_bounds,
                                                   lower_bounds);

    // Initialize Population
    const uint32_t pop_size = DIMENSIONS * AGENTS * ISLES;
    float * const pop_data = gpu_population->_get_dev_transformed_data_array();

    gpu_generator->_generate(pop_size, pop_data);
    gpu_population->_initialize();

    // Define Genome evaluator
    gpu_evaluator = new evaluator_gpu<float>(true,
                                             evaluator_gpu<float>::IGNORE_BOUNDS,
                                             0,
                                             benchmark_gpu_func_1<float>);
  }

  virtual void SetUp() {

    // Common variables allocation
    upper_bounds = new float[DIMENSIONS];
    lower_bounds = new float[DIMENSIONS];

    // Bounds definition
    std::fill(upper_bounds, upper_bounds+DIMENSIONS, 2.0f);
    std::fill(lower_bounds, lower_bounds+DIMENSIONS, 0.0f);

    setupCPU();
    setupGPU();

    // Init timer
    start_time = time(NULL);
  }

  virtual void TearDown() {
    const time_t end_time = time(NULL);
    const time_t elapsed_time = end_time - start_time;

    RecordProperty("Elapsed Time", elapsed_time);
    EXPECT_TRUE(elapsed_time >= 0);

    delete cpu_population;
    delete cpu_generator;
    delete cpu_evaluator;

    delete gpu_population;
    delete gpu_generator;
    delete gpu_evaluator;

    delete [] gpu_data_test_bridge;
    delete [] gpu_fitness_test_bridge;

    delete [] lower_bounds;
    delete [] upper_bounds;
  }

  time_t start_time;

  // Pseudo Random Number Generators
  prngenerator_cpu<float> * cpu_generator;
  prngenerator_gpu<float> * gpu_generator;

  // Evaluator
  evaluator_cpu<float> * cpu_evaluator;
  evaluator_gpu<float> * gpu_evaluator;

  // Population
  population_set_cpu<float> * cpu_population;
  population_set_gpu<float> * gpu_population;

  const uint64_t SEED = 1;
  const size_t GENERATIONS = 1e1;
  const size_t ISLES = 2;
  const size_t AGENTS = 2;
  const size_t DIMENSIONS = 2;

  const size_t fitness_array_size = ISLES * AGENTS;
  const size_t data_array_size = fitness_array_size * DIMENSIONS;

  float * upper_bounds;
  float * lower_bounds;

  float * gpu_fitness_test_bridge;
  float * gpu_data_test_bridge;

};

class GATest : public LocustaTest {
  virtual void SetUp() {
    LocustaTest::SetUp();

    // GPU SOLVER
    // Define solvers
    gpu_solver = new ga_solver_gpu<float>(gpu_population,
                                          gpu_evaluator,
                                          gpu_generator);

    // GA Solver Setup
    gpu_solver->_setup_operators(ga_operators_gpu<float>::tournament_select,
                                 ga_operators_gpu<float>::whole_crossover,
                                 ga_operators_gpu<float>::migration_ring);

    gpu_solver->_set_migration_config(10, // Migration step
                                      1, // Migration size,
                                      4
                                      );

    gpu_solver->_set_selection_config(8, // Selection size,
                                      0.1 // Selection stochastic bias
                                      );

    gpu_solver->_set_breeding_config(0.8, // Crossover rate
                                     0.1  // Mutation rate
                                     );

    gpu_solver->_set_range_extension(0.1);

    gpu_solver->_initialize();

    // BUG: Population must be initialized before solver!
    // gpu_solver->_initialize_population();

    // CPU SOLVER
    // Define solvers
    cpu_solver = new ga_solver_cpu<float>(cpu_population,
                                          cpu_evaluator,
                                          cpu_generator);

    // GA Solver Setup
    cpu_solver->_setup_operators(ga_operators_cpu<float>::tournament_select,
                                 ga_operators_cpu<float>::whole_crossover,
                                 ga_operators_cpu<float>::migration_ring);

    cpu_solver->_set_migration_config(10, // Migration step
                                      1, // Migration size,
                                      4
                                      );

    cpu_solver->_set_selection_config(8, // Selection size,
                                      0.1 // Selection stochastic bias
                                      );

    cpu_solver->_set_breeding_config(0.8, // Crossover rate
                                     0.1  // Mutation rate
                                     );

    cpu_solver->_set_range_extension(0.1);

    cpu_solver->_initialize();

  }

  virtual void TearDown() {
    delete cpu_solver;
    delete gpu_solver;
    LocustaTest::TearDown();
  }

public:
  // GA Solver
  ga_solver_cpu<float> * cpu_solver;
  ga_solver_gpu<float> * gpu_solver;

};

// Test that initialized GPU population's genome values are valid.
TEST_F(GATest, GPUBoundedGenomeValues) {
  gpu_population->_copy_dev_data_into_host(gpu_data_test_bridge);
  for (uint32_t i = 0; i < data_array_size; ++i)
    {
      uint32_t mapped_dim = i / (ISLES * AGENTS);
      EXPECT_LE(gpu_data_test_bridge[i], upper_bounds[mapped_dim]);
      EXPECT_GE(gpu_data_test_bridge[i], lower_bounds[mapped_dim]);
    }
}

// Test that GPU population's fitness values are initialized to -inf.
TEST_F(GATest, GPUInitFitnessValues) {
  gpu_population->_copy_dev_fitness_into_host(gpu_fitness_test_bridge);
  for (uint32_t i = 0; i < fitness_array_size; ++i)
    {
      EXPECT_EQ(gpu_fitness_test_bridge[i], -std::numeric_limits<float>::infinity());
    }
}

// Test that initialized CPU population's genome values are valid.
TEST_F(GATest, CPUBoundedGenomeValues) {
  float * cpu_data_test_bridge = cpu_population->_get_data_array();
  for (uint32_t i = 0; i < data_array_size; ++i)
    {
      uint32_t mapped_dim = i / (ISLES * AGENTS);
      EXPECT_LE(cpu_data_test_bridge[i], upper_bounds[mapped_dim]);
      EXPECT_GE(cpu_data_test_bridge[i], lower_bounds[mapped_dim]);
    }
}

// Test that CPU population's fitness values are initialized to -inf.
TEST_F(GATest, CPUInitFitnessValues) {
  float * cpu_fitness_test_bridge = cpu_population->_get_fitness_array();
  for (uint32_t i = 0; i < fitness_array_size; ++i)
    {
      EXPECT_EQ(cpu_fitness_test_bridge[i], -std::numeric_limits<float>::infinity());
    }
}


// // Run the GPU solver
// TEST_F(GATest, GpuSolverRun) {

//     for(size_t g = 0; g < GENERATIONS; ++g)
//     {
// #ifdef _DEBUG
//         std::cout << "Generation: " << g << std::endl;
// #else
//         std::cout << "Generation: " << g << "\r";
// #endif
//         gpu_solver->_advance_generation();
//         //gpu_solver->_print_gpu_solver_elite();
//     }
//     gpu_solver->_print_solver_solution();
// }

// // Run the CPU solver
// TEST_F(GATest, CpuSolverRun) {

//     for(size_t g = 0; g < GENERATIONS; ++g)
//     {
// #ifdef _DEBUG
//         std::cout << "Generation: " << g << std::endl;
// #else
//         std::cout << "Generation: " << g << "\r";
// #endif
//         cpu_solver->_advance_generation();
//         //cpu_solver->_print_cpu_solver_elite();
//     }
//     cpu_solver->_print_solver_solution();
// }

// Test that CPU vs GPU evaluation values.
TEST_F(GATest, CompareFitnessValues) {

  // Sync implementation data
  float * cpu_genomes = cpu_population->_get_data_array();
  gpu_population->_copy_dev_data_into_host(gpu_data_test_bridge);
  // Reorganize genomes
  for (uint32_t i = 0; i < ISLES; ++i)
    {
      for(uint32_t j = 0; j < AGENTS; ++j)
        {
          for(uint32_t k = 0; k < DIMENSIONS; ++k)
            {
              uint32_t locus_offset = k * ISLES * AGENTS;
              cpu_genomes[i * AGENTS * DIMENSIONS + j * DIMENSIONS + k] = gpu_data_test_bridge[locus_offset + i * AGENTS + j];
            }
        }
    }

  // Evaluate Genomes
  gpu_solver->_evaluate_genomes();
  cpu_solver->_evaluate_genomes();

  // Compare Fitness Evaluation
  gpu_population->_copy_dev_fitness_into_host(gpu_fitness_test_bridge);
  float * cpu_fitness_test_bridge = cpu_population->_get_fitness_array();

  for (uint32_t i = 0; i < fitness_array_size; ++i)
    {
      const float diff = gpu_fitness_test_bridge[i] - cpu_fitness_test_bridge[i];
      const float tolerance = 1e-4f;
      EXPECT_LE(diff, tolerance);
    }
}
