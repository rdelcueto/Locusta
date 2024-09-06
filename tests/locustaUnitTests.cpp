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

std::vector<uint32_t> BenchmarkFunctions;
std::vector<uint32_t> IslesCombinations;
std::vector<uint32_t> AgentsCombinations;
std::vector<uint32_t> DimensionsCombinations;

const uint32_t GLOBAL_SEED = 314;
const uint32_t GLOBAL_EVALUATIONS = (128*128*128)*2e1;

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

int main(int argc, char **argv) {
  // Parallel Compare
  BenchmarkFunctions = {1, 9};

  IslesCombinations = {1, 4, 8, 16};

  AgentsCombinations = {32, 64, 128, 256, 512};

  DimensionsCombinations = {128, 256, 512, 1024, 2048};

  // CPU Cache Assoc test
  // BenchmarkFunctions = {1, 9};
  // IslesCombinations = {4};
  // AgentsCombinations = {64, 128, 256};

  // DimensionsCombinations = {
  //   16,   24,   32,   40,   48,   56,   64,   72,   80,   88,   96,
  //   104,  112,  120,  128,  136,  144,  152,  160,  168,  176,  184,
  //   192,  200,  208,  216,  224,  232,  240,  248,  256,  264,  272,
  //   280,  288,  296,  304,  312,  320,  328,  336,  344,  352,  360,
  //   368,  376,  384,  392,  400,  408,  416,  424,  432,  440,  448,
  //   456,  464,  472,  480,  488,  496,  504,  512,  520,  528,  536,
  //   544,  552,  560,  568,  576,  584,  592,  600,  608,  616,  624,
  //   632,  640,  648,  656,  664,  672,  680,  688,  696,  704,  712,
  //   720,  728,  736,  744,  752,  760,  768,  776,  784,  792,  800,
  //   808,  816,  824,  832,  840,  848,  856,  864,  872,  880,  888,
  //   896,  904,  912,  920,  928,  936,  944,  952,  960,  968,  976,
  //   984,  992, 1000, 1008, 1016, 1024 };

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
