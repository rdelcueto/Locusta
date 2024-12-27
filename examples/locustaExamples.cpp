#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <tuple>
#include <time.h>

#include "./benchmarks/benchmarks_cpu.hpp"
#include "./benchmarks/benchmarks_cuda.hpp"

#include "evaluator/evaluator_cpu.hpp"
#include "evaluator/evaluator_cuda.hpp"

#include "prngenerator/prngenerator_cpu.hpp"
#include "prngenerator/prngenerator_cuda.hpp"

#include "solvers/pso/pso_solver_cpu.hpp"
#include "solvers/pso/pso_solver_cuda.hpp"

#include "solvers/pso/pso_operators/pso_std_operators_cpu_impl.hpp"
#include "solvers/pso/pso_operators/pso_std_operators_cuda_impl.hpp"

#include "solvers/ga/ga_solver_cpu.hpp"
#include "solvers/ga/ga_solver_cuda.hpp"

#include "solvers/ga/ga_operators/ga_std_operators_cpu_impl.hpp"
#include "solvers/ga/ga_operators/ga_std_operators_cuda_impl.hpp"

#include "solvers/de/de_solver_cpu.hpp"
#include "solvers/de/de_solver_cuda.hpp"

#include "solvers/de/de_operators/de_std_operators_cpu_impl.hpp"
#include "solvers/de/de_operators/de_std_operators_cuda_impl.hpp"

#include "cuda_runtime.h"

using namespace locusta;

/**
 * @brief Vector of benchmark function identifiers.
 */
std::vector<uint32_t> BenchmarkFunctions;

/**
 * @brief Vector of isle combinations.
 */
std::vector<uint32_t> IslesCombinations;

/**
 * @brief Vector of agent combinations.
 */
std::vector<uint32_t> AgentsCombinations;

/**
 * @brief Vector of dimension combinations.
 */
std::vector<uint32_t> DimensionsCombinations;

/**
 * @brief Global seed for random number generation.
 */
const uint32_t GLOBAL_SEED = 314;

/**
 * @brief Global number of evaluations.
 */
const uint32_t GLOBAL_EVALUATIONS = (128*128*128)*2e1;

/**
 * @brief Test fixture for CPU-based Locusta tests.
 *
 * This class provides a test fixture for CPU-based Locusta tests,
 * allowing for parameterized testing with different combinations of
 * benchmark function identifiers, isles, agents, and dimensions.
 */
class CPULocustaTest : public testing::TestWithParam<
  std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>
{

public:
  virtual ~CPULocustaTest() {}

protected:
  virtual void SetUp()
  {
    GENERATIONS = GENERATIONS != 0 ? GENERATIONS : 1;

    upper_bounds_ptr = new float[DIMENSIONS];
    lower_bounds_ptr = new float[DIMENSIONS];

    // Bounds definition
    std::fill(upper_bounds_ptr, upper_bounds_ptr + DIMENSIONS, 100.0f);
    std::fill(lower_bounds_ptr, lower_bounds_ptr + DIMENSIONS, -100.0f);

    evaluation_functor_cpu_ptr =
      new BenchmarkFunctor<float>(BENCHMARK_FUNC_ID, DIMENSIONS);

    evaluator_cpu_ptr = new evaluator_cpu<float>(
      evaluation_functor_cpu_ptr, true, BoundMapKind::CropBounds, DIMENSIONS);

    // Single prng generator for CPU solver seems to perform better.
    // prngenerator_cpu_ptr = new prngenerator_cpu<float>(1);
    prngenerator_cpu_ptr = new prngenerator_cpu<float>(omp_get_max_threads());
    prngenerator_cpu_ptr->_initialize_engines(SEED);

    population_cpu_ptr =
      new population_set_cpu<float>(ISLES, AGENTS, DIMENSIONS);

    std::cout << "FuncId: " << BENCHMARK_FUNC_ID
              << ", Generations: " << GENERATIONS
              << ", Isles: " << ISLES
              << ", Agents: " << AGENTS
              << ", Dimensions: " << DIMENSIONS
              << ", CV: " << (GENERATIONS * ISLES * AGENTS * DIMENSIONS) << std::endl;
  }

  virtual void TearDown()
  {
    delete population_cpu_ptr;
    population_cpu_ptr = NULL;

    delete prngenerator_cpu_ptr;
    prngenerator_cpu_ptr = NULL;

    delete evaluator_cpu_ptr;
    evaluator_cpu_ptr = NULL;

    delete evaluation_functor_cpu_ptr;
    evaluation_functor_cpu_ptr = NULL;

    delete [] lower_bounds_ptr;
    lower_bounds_ptr = NULL;

    delete [] upper_bounds_ptr;
    upper_bounds_ptr = NULL;
  }

public:
  const uint64_t SEED = GLOBAL_SEED;

  uint32_t const BENCHMARK_FUNC_ID = std::get<0>(GetParam());
  uint32_t const ISLES = std::get<1>(GetParam());
  uint32_t const AGENTS = std::get<2>(GetParam());
  uint32_t const DIMENSIONS = std::get<3>(GetParam());

  uint32_t const FUNC_WEIGHT = BENCHMARK_FUNC_ID == 1 ? 1e1 : 2e0;
  uint64_t GENERATIONS = (FUNC_WEIGHT * GLOBAL_EVALUATIONS) / (ISLES * AGENTS * DIMENSIONS);

  float* upper_bounds_ptr;
  float* lower_bounds_ptr;

  // Pseudo Random Number Generators
  prngenerator_cpu<float>* prngenerator_cpu_ptr;
  // Evaluator
  evaluator_cpu<float>* evaluator_cpu_ptr;
  EvaluationFunctor<float>* evaluation_functor_cpu_ptr;
  // Population
  population_set_cpu<float>* population_cpu_ptr;
};

/**
 * @brief Test fixture for GPU-based Locusta tests.
 *
 * This class provides a test fixture for GPU-based Locusta tests,
 * allowing for parameterized testing with different combinations of
 * benchmark function identifiers, isles, agents, and dimensions.
 */
class GPULocustaTest : public testing::TestWithParam<
  std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>
{

public:
  virtual ~GPULocustaTest() {}

protected:
  virtual void SetUp()
  {
    GENERATIONS = GENERATIONS != 0 ? GENERATIONS : 1;

    __setup_cuda(0, 1);

    upper_bounds_ptr = new float[DIMENSIONS];
    lower_bounds_ptr = new float[DIMENSIONS];

    // Bounds definition
    std::fill(upper_bounds_ptr, upper_bounds_ptr + DIMENSIONS, 100.0f);
    std::fill(lower_bounds_ptr, lower_bounds_ptr + DIMENSIONS, -100.0f);

    evaluation_functor_cuda_ptr =
      new BenchmarkCudaFunctor<float>(BENCHMARK_FUNC_ID, DIMENSIONS);

    evaluator_cuda_ptr = new evaluator_cuda<float>(
      evaluation_functor_cuda_ptr, true, BoundMapKind::CropBounds, DIMENSIONS);

    prngenerator_cuda_ptr = new prngenerator_cuda<float>(ISLES * AGENTS);
    prngenerator_cuda_ptr->_initialize_engines(SEED);

    population_cuda_ptr =
      new population_set_cuda<float>(ISLES, AGENTS, DIMENSIONS);

    std::cout << "FuncId: " << BENCHMARK_FUNC_ID
              << ", Generations: " << GENERATIONS
              << ", Isles: " << ISLES
              << ", Agents: " << AGENTS
              << ", Dimensions: " << DIMENSIONS
              << ", CV: " << (GENERATIONS * ISLES * AGENTS * DIMENSIONS) << std::endl;
  }

  virtual void TearDown()
  {
    delete population_cuda_ptr;
    population_cuda_ptr = NULL;

    delete prngenerator_cuda_ptr;
    prngenerator_cuda_ptr = NULL;

    delete evaluator_cuda_ptr;
    evaluator_cuda_ptr = NULL;

    delete evaluation_functor_cuda_ptr;
    evaluation_functor_cuda_ptr = NULL;

    delete [] lower_bounds_ptr;
    lower_bounds_ptr = NULL;

    delete [] upper_bounds_ptr;
    upper_bounds_ptr = NULL;
  }

public:
  const uint64_t SEED = GLOBAL_SEED;

  uint32_t const BENCHMARK_FUNC_ID = std::get<0>(GetParam());
  uint32_t const ISLES = std::get<1>(GetParam());
  uint32_t const AGENTS = std::get<2>(GetParam());
  uint32_t const DIMENSIONS = std::get<3>(GetParam());

  uint32_t const GPU_FACTOR = 1e0;
  uint32_t const FUNC_WEIGHT = BENCHMARK_FUNC_ID == 1 ? 1e1 : 2e0;
  uint64_t GENERATIONS = (FUNC_WEIGHT * GLOBAL_EVALUATIONS) / (ISLES * AGENTS * DIMENSIONS);

  float* upper_bounds_ptr;
  float* lower_bounds_ptr;

  // Pseudo Random Number Generators
  prngenerator_cuda<float>* prngenerator_cuda_ptr;
  // Evaluator
  EvaluationCudaFunctor<float>* evaluation_functor_cuda_ptr;
  evaluator_cuda<float>* evaluator_cuda_ptr;
  // Population
  population_set_cuda<float>* population_cuda_ptr;
};

/**
 * @brief Test fixture for CPU-based Particle Swarm Optimization tests.
 *
 * This class provides a test fixture for CPU-based Particle Swarm Optimization tests,
 * inheriting from the CPULocustaTest class and adding a PSO solver.
 */
class CPUParticleSwarmTest : public CPULocustaTest
{

public:
  virtual ~CPUParticleSwarmTest() {}

protected:
  virtual void SetUp()
  {
    CPULocustaTest::SetUp();
    pso_solver_cpu_ptr = new pso_solver_cpu<float>(
      population_cpu_ptr, evaluator_cpu_ptr, prngenerator_cpu_ptr, GENERATIONS,
      upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown()
  {
    delete pso_solver_cpu_ptr;
    pso_solver_cpu_ptr = NULL;
    CPULocustaTest::TearDown();
  }

public:
  pso_solver_cpu<float>* pso_solver_cpu_ptr;
};

/**
 * @brief Test fixture for GPU-based Particle Swarm Optimization tests.
 *
 * This class provides a test fixture for GPU-based Particle Swarm Optimization tests,
 * inheriting from the GPULocustaTest class and adding a PSO solver.
 */
class GPUParticleSwarmTest : public GPULocustaTest
{

public:
  virtual ~GPUParticleSwarmTest() {}

protected:
  virtual void SetUp()
  {
    GPULocustaTest::SetUp();
    pso_solver_cuda_ptr = new pso_solver_cuda<float>(
      population_cuda_ptr, evaluator_cuda_ptr, prngenerator_cuda_ptr,
      GENERATIONS, upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown()
  {
    delete pso_solver_cuda_ptr;
    pso_solver_cuda_ptr = NULL;
    GPULocustaTest::TearDown();
  }

public:
  pso_solver_cuda<float>* pso_solver_cuda_ptr;
};

/**
 * @brief Test fixture for CPU-based Genetic Algorithm tests.
 *
 * This class provides a test fixture for CPU-based Genetic Algorithm tests,
 * inheriting from the CPULocustaTest class and adding a GA solver.
 */
class CPUGeneticAlgorithmTest : public CPULocustaTest
{
public:
  virtual ~CPUGeneticAlgorithmTest() {}

protected:
  virtual void SetUp()
  {
    CPULocustaTest::SetUp();
    ga_solver_cpu_ptr = new ga_solver_cpu<float>(
      population_cpu_ptr, evaluator_cpu_ptr, prngenerator_cpu_ptr, GENERATIONS,
      upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown()
  {
    delete ga_solver_cpu_ptr;
    ga_solver_cpu_ptr = NULL;
    CPULocustaTest::TearDown();
  }

public:
  ga_solver_cpu<float>* ga_solver_cpu_ptr;
};

/**
 * @brief Test fixture for GPU-based Genetic Algorithm tests.
 *
 * This class provides a test fixture for GPU-based Genetic Algorithm tests,
 * inheriting from the GPULocustaTest class and adding a GA solver.
 */
class GPUGeneticAlgorithmTest : public GPULocustaTest
{
public:
  virtual ~GPUGeneticAlgorithmTest() {}

protected:
  virtual void SetUp()
  {
    GPULocustaTest::SetUp();
    ga_solver_cuda_ptr = new ga_solver_cuda<float>(
      population_cuda_ptr, evaluator_cuda_ptr, prngenerator_cuda_ptr,
      GENERATIONS, upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown()
  {
    delete ga_solver_cuda_ptr;
    ga_solver_cuda_ptr = NULL;
    GPULocustaTest::TearDown();
  }

public:
  ga_solver_cuda<float>* ga_solver_cuda_ptr;
};

/**
 * @brief Test fixture for CPU-based Differential Evolution tests.
 *
 * This class provides a test fixture for CPU-based Differential Evolution tests,
 * inheriting from the CPULocustaTest class and adding a DE solver.
 */
class CPUDifferentialEvolutionTest : public CPULocustaTest
{
public:
  virtual ~CPUDifferentialEvolutionTest() {}

protected:
  virtual void SetUp()
  {
    CPULocustaTest::SetUp();
    de_solver_cpu_ptr = new de_solver_cpu<float>(
      population_cpu_ptr, evaluator_cpu_ptr, prngenerator_cpu_ptr, GENERATIONS,
      upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown()
  {
    delete de_solver_cpu_ptr;
    de_solver_cpu_ptr = NULL;
    CPULocustaTest::TearDown();
  }

public:
  de_solver_cpu<float>* de_solver_cpu_ptr;
};

/**
 * @brief Test fixture for GPU-based Differential Evolution tests.
 *
 * This class provides a test fixture for GPU-based Differential Evolution tests,
 * inheriting from the GPULocustaTest class and adding a DE solver.
 */
class GPUDifferentialEvolutionTest : public GPULocustaTest
{
public:
  virtual ~GPUDifferentialEvolutionTest() {}

protected:
  virtual void SetUp()
  {
    GPULocustaTest::SetUp();
    de_solver_cuda_ptr = new de_solver_cuda<float>(
      population_cuda_ptr, evaluator_cuda_ptr, prngenerator_cuda_ptr,
      GENERATIONS, upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown()
  {
    delete de_solver_cuda_ptr;
    de_solver_cuda_ptr = NULL;
    GPULocustaTest::TearDown();
  }

public:
  de_solver_cuda<float>* de_solver_cuda_ptr;
};

/**
 * @brief Test case for CPU-based Particle Swarm Optimization.
 */
TEST_P(CPUParticleSwarmTest, BenchmarkCpu)
{
  CanonicalParticleRecordUpdate<float> cpru;
  CanonicalSpeedUpdate<float> csu;
  CanonicalPositionUpdate<float> cpu;

  pso_solver_cpu_ptr->setup_operators(&cpru, &csu, &cpu);

  pso_solver_cpu_ptr->setup_solver();
  pso_solver_cpu_ptr->run();
  // pso_solver_cpu_ptr->print_solutions();
  pso_solver_cpu_ptr->teardown_solver();
}

/**
 * @brief Test case for CPU-based Genetic Algorithm.
 */
TEST_P(CPUGeneticAlgorithmTest, BenchmarkCpu)
{
  WholeCrossover<float> wc;
  TournamentSelection<float> ts;

  ga_solver_cpu_ptr->setup_operators(&wc, &ts);

  ga_solver_cpu_ptr->setup_solver();
  ga_solver_cpu_ptr->run();
  // ga_solver_cpu_ptr->print_solutions();
  ga_solver_cpu_ptr->teardown_solver();
}

/**
 * @brief Test case for CPU-based Differential Evolution.
 */
TEST_P(CPUDifferentialEvolutionTest, BenchmarkCpu)
{
  DeWholeCrossover<float> dwc;
  DeRandomSelection<float> drs;

  de_solver_cpu_ptr->setup_operators(&dwc, &drs);

  de_solver_cpu_ptr->setup_solver();
  de_solver_cpu_ptr->run();
  // de_solver_cpu_ptr->print_solutions();
  de_solver_cpu_ptr->teardown_solver();
}

/**
 * @brief Test case for GPU-based Particle Swarm Optimization.
 */
TEST_P(GPUParticleSwarmTest, BenchmarkCuda)
{
  CanonicalParticleRecordUpdateCuda<float> cpru;
  CanonicalSpeedUpdateCuda<float> csu;
  CanonicalPositionUpdateCuda<float> cpu;

  pso_solver_cuda_ptr->setup_operators(&cpru, &csu, &cpu);

  pso_solver_cuda_ptr->setup_solver();
  pso_solver_cuda_ptr->run();
  // pso_solver_cuda_ptr->print_population();
  pso_solver_cuda_ptr->teardown_solver();
}

/**
 * @brief Test fixture for GPU genetic algorithm.
 */
TEST_P(GPUGeneticAlgorithmTest, BenchmarkCuda)
{
  WholeCrossoverCuda<float> wc;
  TournamentSelectionCuda<float> ts;
  ga_solver_cuda_ptr->setup_operators(&wc, &ts);

  ga_solver_cuda_ptr->setup_solver();
  ga_solver_cuda_ptr->run();
  // ga_solver_cuda_ptr->print_population();
  ga_solver_cuda_ptr->teardown_solver();
}

/**
 * @brief Test fixture for GPU differential evolution.
 */
TEST_P(GPUDifferentialEvolutionTest, BenchmarkCuda)
{
  DeWholeCrossoverCuda<float> dwc;
  DeRandomSelectionCuda<float> drs;
  de_solver_cuda_ptr->setup_operators(&dwc, &drs);

  de_solver_cuda_ptr->setup_solver();
  de_solver_cuda_ptr->run();
  // de_solver_cuda_ptr->print_population();
  de_solver_cuda_ptr->teardown_solver();
}

// CPU Benchmarks
INSTANTIATE_TEST_CASE_P(CPUParticleSwarmTestSuite,
                        CPUParticleSwarmTest,
                        ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                                           ::testing::ValuesIn(IslesCombinations),
                                           ::testing::ValuesIn(AgentsCombinations),
                                           ::testing::ValuesIn(DimensionsCombinations)
                                           ));

INSTANTIATE_TEST_CASE_P(CPUGeneticAlgorithmTestSuite,
                        CPUGeneticAlgorithmTest,
                        ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                                           ::testing::ValuesIn(IslesCombinations),
                                           ::testing::ValuesIn(AgentsCombinations),
                                           ::testing::ValuesIn(DimensionsCombinations)
                                           ));

INSTANTIATE_TEST_CASE_P(CPUDifferentialEvolutionTestSuite,
                        CPUDifferentialEvolutionTest,
                        ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                                           ::testing::ValuesIn(IslesCombinations),
                                           ::testing::ValuesIn(AgentsCombinations),
                                           ::testing::ValuesIn(DimensionsCombinations)
                                           ));

// GPU Benchmarks
INSTANTIATE_TEST_CASE_P(GPUParticleSwarmTestSuite,
                        GPUParticleSwarmTest,
                        ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                                           ::testing::ValuesIn(IslesCombinations),
                                           ::testing::ValuesIn(AgentsCombinations),
                                           ::testing::ValuesIn(DimensionsCombinations)
                                           ));

INSTANTIATE_TEST_CASE_P(GPUGeneticAlgorithmTestSuite,
                        GPUGeneticAlgorithmTest,
                        ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                                           ::testing::ValuesIn(IslesCombinations),
                                           ::testing::ValuesIn(AgentsCombinations),
                                           ::testing::ValuesIn(DimensionsCombinations)
                                           ));

INSTANTIATE_TEST_CASE_P(GPUDifferentialEvolutionTestSuite,
                        GPUDifferentialEvolutionTest,
                        ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                                           ::testing::ValuesIn(IslesCombinations),
                                           ::testing::ValuesIn(AgentsCombinations),
                                           ::testing::ValuesIn(DimensionsCombinations)
                                           ));

/**
 * @brief Main function for running the unit tests.
 * 
 * This function initializes the Google Test framework and runs all the tests.
 * 
 * The following parameters are used to configure the tests:
 * 
 *  - BenchmarkFunctions: A list of benchmark function identifiers to test.
 *  - IslesCombinations: A list of isle combinations to test.
 *  - AgentsCombinations: A list of agent combinations to test.
 *  - DimensionsCombinations: A list of dimension combinations to test.
 * 
 * @param argc Number of command line arguments.
 * @param argv Array of command line arguments.
 * @return 0 if all tests pass, otherwise 1.
 */
int main(int argc, char **argv) {
  // Parallel Compare
  BenchmarkFunctions = {1, 9};

  IslesCombinations = {1, 4, 8, 16};

  AgentsCombinations = {32, 64, 128, 256, 512};

  DimensionsCombinations = {128, 256, 512, 1024, 2048};

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
