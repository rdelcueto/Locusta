#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <time.h>
#include <tuple>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// CPU Includes
#include "./benchmarks/benchmarks_cpu.hpp"
#include "evaluator/evaluator_cpu.hpp"
#include "prngenerator/prngenerator_cpu.hpp"
#include "solvers/de/de_operators/de_std_operators_cpu_impl.hpp"
#include "solvers/de/de_solver_cpu.hpp"
#include "solvers/ga/ga_operators/ga_std_operators_cpu_impl.hpp"
#include "solvers/ga/ga_solver_cpu.hpp"
#include "solvers/pso/pso_operators/pso_std_operators_cpu_impl.hpp"
#include "solvers/pso/pso_solver_cpu.hpp"

using namespace locusta;

std::vector<uint32_t> BenchmarkFunctions;
std::vector<uint32_t> IslesCombinations;
std::vector<uint32_t> AgentsCombinations;
std::vector<uint32_t> DimensionsCombinations;

const uint32_t GLOBAL_SEED = 314;
const uint32_t GLOBAL_EVALUATIONS = (64 * 64 * 64) * 1e2;

// =================================================================================================
// Standard Single Precision CPU Tests
// =================================================================================================

class CPULocustaTest : public testing::TestWithParam<
                           std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> {

public:
  virtual ~CPULocustaTest() {}

protected:
  virtual void SetUp() {
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

    prngenerator_cpu_ptr = new prngenerator_cpu<float>(omp_get_max_threads());
    prngenerator_cpu_ptr->_initialize_engines(SEED);

    population_cpu_ptr =
        new population_set_cpu<float>(ISLES, AGENTS, DIMENSIONS);

    std::cout << "FuncId: " << BENCHMARK_FUNC_ID
              << ", Generations: " << GENERATIONS << ", Isles: " << ISLES
              << ", Agents: " << AGENTS << ", Dimensions: " << DIMENSIONS
              << ", CV: " << (GENERATIONS * ISLES * AGENTS * DIMENSIONS)
              << std::endl;
  }

  virtual void TearDown() {
    delete population_cpu_ptr;
    population_cpu_ptr = NULL;

    delete prngenerator_cpu_ptr;
    prngenerator_cpu_ptr = NULL;

    delete evaluator_cpu_ptr;
    evaluator_cpu_ptr = NULL;

    delete evaluation_functor_cpu_ptr;
    evaluation_functor_cpu_ptr = NULL;

    delete[] lower_bounds_ptr;
    lower_bounds_ptr = NULL;

    delete[] upper_bounds_ptr;
    upper_bounds_ptr = NULL;
  }

public:
  const uint64_t SEED = GLOBAL_SEED;

  uint32_t const BENCHMARK_FUNC_ID = std::get<0>(GetParam());
  uint32_t const ISLES = std::get<1>(GetParam());
  uint32_t const AGENTS = std::get<2>(GetParam());
  uint32_t const DIMENSIONS = std::get<3>(GetParam());

  uint32_t const FUNC_WEIGHT = BENCHMARK_FUNC_ID == 1 ? 1e1 : 2e0;
  uint64_t GENERATIONS =
      (FUNC_WEIGHT * GLOBAL_EVALUATIONS) / (ISLES * AGENTS * DIMENSIONS);

  float *upper_bounds_ptr;
  float *lower_bounds_ptr;

  prngenerator_cpu<float> *prngenerator_cpu_ptr;
  evaluator_cpu<float> *evaluator_cpu_ptr;
  EvaluationFunctor<float> *evaluation_functor_cpu_ptr;
  population_set_cpu<float> *population_cpu_ptr;
};

class CPUParticleSwarmTest : public CPULocustaTest {
public:
  virtual ~CPUParticleSwarmTest() {}

protected:
  virtual void SetUp() {
    CPULocustaTest::SetUp();
    pso_solver_cpu_ptr = new pso_solver_cpu<float>(
        population_cpu_ptr, evaluator_cpu_ptr, prngenerator_cpu_ptr,
        GENERATIONS, upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown() {
    delete pso_solver_cpu_ptr;
    pso_solver_cpu_ptr = NULL;
    CPULocustaTest::TearDown();
  }

public:
  pso_solver_cpu<float> *pso_solver_cpu_ptr;
};

class CPUGeneticAlgorithmTest : public CPULocustaTest {
public:
  virtual ~CPUGeneticAlgorithmTest() {}

protected:
  virtual void SetUp() {
    CPULocustaTest::SetUp();
    ga_solver_cpu_ptr = new ga_solver_cpu<float>(
        population_cpu_ptr, evaluator_cpu_ptr, prngenerator_cpu_ptr,
        GENERATIONS, upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown() {
    delete ga_solver_cpu_ptr;
    ga_solver_cpu_ptr = NULL;
    CPULocustaTest::TearDown();
  }

public:
  ga_solver_cpu<float> *ga_solver_cpu_ptr;
};

class CPUDifferentialEvolutionTest : public CPULocustaTest {
public:
  virtual ~CPUDifferentialEvolutionTest() {}

protected:
  virtual void SetUp() {
    CPULocustaTest::SetUp();
    de_solver_cpu_ptr = new de_solver_cpu<float>(
        population_cpu_ptr, evaluator_cpu_ptr, prngenerator_cpu_ptr,
        GENERATIONS, upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown() {
    delete de_solver_cpu_ptr;
    de_solver_cpu_ptr = NULL;
    CPULocustaTest::TearDown();
  }

public:
  de_solver_cpu<float> *de_solver_cpu_ptr;
};

TEST_P(CPUParticleSwarmTest, BenchmarkCpu) {
  CanonicalParticleRecordUpdate<float> cpru;
  CanonicalSpeedUpdate<float> csu;
  CanonicalPositionUpdate<float> cpu;

  pso_solver_cpu_ptr->setup_operators(&cpru, &csu, &cpu);
  pso_solver_cpu_ptr->setup_solver();
  pso_solver_cpu_ptr->run();
  pso_solver_cpu_ptr->teardown_solver();
}

TEST_P(CPUGeneticAlgorithmTest, BenchmarkCpu) {
  WholeCrossover<float> wc;
  TournamentSelection<float> ts;

  ga_solver_cpu_ptr->setup_operators(&wc, &ts);
  ga_solver_cpu_ptr->setup_solver();
  ga_solver_cpu_ptr->run();
  ga_solver_cpu_ptr->teardown_solver();
}

TEST_P(CPUDifferentialEvolutionTest, BenchmarkCpu) {
  DeWholeCrossover<float> dwc;
  DeRandomSelection<float> drs;

  de_solver_cpu_ptr->setup_operators(&dwc, &drs);
  de_solver_cpu_ptr->setup_solver();
  de_solver_cpu_ptr->run();
  de_solver_cpu_ptr->teardown_solver();
}

INSTANTIATE_TEST_CASE_P(
    CPUParticleSwarmTestSuite, CPUParticleSwarmTest,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    CPUGeneticAlgorithmTestSuite, CPUGeneticAlgorithmTest,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    CPUDifferentialEvolutionTestSuite, CPUDifferentialEvolutionTest,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

// =================================================================================================
// Half & Quarter Precision CPU Tests
// =================================================================================================

template <typename TFloat>
class CPULocustaPrecisionTest
    : public testing::TestWithParam<
          std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> {
public:
  virtual ~CPULocustaPrecisionTest() {}

protected:
  virtual void SetUp() {
    GENERATIONS = GENERATIONS != 0 ? GENERATIONS : 1;

    upper_bounds_ptr = new TFloat[DIMENSIONS];
    lower_bounds_ptr = new TFloat[DIMENSIONS];

    for (uint32_t i = 0; i < DIMENSIONS; ++i) {
      upper_bounds_ptr[i] = static_cast<TFloat>(100.0f);
      lower_bounds_ptr[i] = static_cast<TFloat>(-100.0f);
    }

    evaluation_functor_cpu_ptr =
        new BenchmarkFunctor<TFloat>(BENCHMARK_FUNC_ID, DIMENSIONS);

    evaluator_cpu_ptr = new evaluator_cpu<TFloat>(
        evaluation_functor_cpu_ptr, true, BoundMapKind::CropBounds, DIMENSIONS);

    prngenerator_cpu_ptr = new prngenerator_cpu<TFloat>(omp_get_max_threads());
    prngenerator_cpu_ptr->_initialize_engines(SEED);

    population_cpu_ptr =
        new population_set_cpu<TFloat>(ISLES, AGENTS, DIMENSIONS);

    std::cout << "[CPU Low-Precision Test] Type size: " << sizeof(TFloat) << "B"
              << ", FuncId: " << BENCHMARK_FUNC_ID << ", Isles: " << ISLES
              << ", Agents: " << AGENTS << ", Dimensions: " << DIMENSIONS
              << std::endl;
  }

  virtual void TearDown() {
    delete population_cpu_ptr;
    population_cpu_ptr = NULL;

    delete prngenerator_cpu_ptr;
    prngenerator_cpu_ptr = NULL;

    delete evaluator_cpu_ptr;
    evaluator_cpu_ptr = NULL;

    delete evaluation_functor_cpu_ptr;
    evaluation_functor_cpu_ptr = NULL;

    delete[] lower_bounds_ptr;
    lower_bounds_ptr = NULL;

    delete[] upper_bounds_ptr;
    upper_bounds_ptr = NULL;
  }

public:
  const uint64_t SEED = GLOBAL_SEED;

  uint32_t const BENCHMARK_FUNC_ID = std::get<0>(GetParam());
  uint32_t const ISLES = std::get<1>(GetParam());
  uint32_t const AGENTS = std::get<2>(GetParam());
  uint32_t const DIMENSIONS = std::get<3>(GetParam());

  uint32_t const FUNC_WEIGHT = BENCHMARK_FUNC_ID == 1 ? 1e1 : 2e0;
  uint64_t GENERATIONS =
      (FUNC_WEIGHT * GLOBAL_EVALUATIONS) / (ISLES * AGENTS * DIMENSIONS);

  TFloat *upper_bounds_ptr;
  TFloat *lower_bounds_ptr;

  prngenerator_cpu<TFloat> *prngenerator_cpu_ptr;
  evaluator_cpu<TFloat> *evaluator_cpu_ptr;
  EvaluationFunctor<TFloat> *evaluation_functor_cpu_ptr;
  population_set_cpu<TFloat> *population_cpu_ptr;
};

class CPUHalfPrecisionTest : public CPULocustaPrecisionTest<__half> {};
class CPUBFloat16Test : public CPULocustaPrecisionTest<__nv_bfloat16> {};
class CPUFP8E4M3Test : public CPULocustaPrecisionTest<__nv_fp8_e4m3> {};
class CPUFP8E5M2Test : public CPULocustaPrecisionTest<__nv_fp8_e5m2> {};

TEST_P(CPUHalfPrecisionTest, PSO_Half) {
  CanonicalParticleRecordUpdate<__half> cpru;
  CanonicalSpeedUpdate<__half> csu;
  CanonicalPositionUpdate<__half> cpu;

  pso_solver_cpu<__half> solver(population_cpu_ptr, evaluator_cpu_ptr,
                                prngenerator_cpu_ptr, GENERATIONS,
                                upper_bounds_ptr, lower_bounds_ptr);
  solver.setup_operators(&cpru, &csu, &cpu);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

TEST_P(CPUBFloat16Test, PSO_BFloat16) {
  CanonicalParticleRecordUpdate<__nv_bfloat16> cpru;
  CanonicalSpeedUpdate<__nv_bfloat16> csu;
  CanonicalPositionUpdate<__nv_bfloat16> cpu;

  pso_solver_cpu<__nv_bfloat16> solver(population_cpu_ptr, evaluator_cpu_ptr,
                                       prngenerator_cpu_ptr, GENERATIONS,
                                       upper_bounds_ptr, lower_bounds_ptr);
  solver.setup_operators(&cpru, &csu, &cpu);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

TEST_P(CPUFP8E4M3Test, PSO_FP8_E4M3) {
  CanonicalParticleRecordUpdate<__nv_fp8_e4m3> cpru;
  CanonicalSpeedUpdate<__nv_fp8_e4m3> csu;
  CanonicalPositionUpdate<__nv_fp8_e4m3> cpu;

  pso_solver_cpu<__nv_fp8_e4m3> solver(population_cpu_ptr, evaluator_cpu_ptr,
                                       prngenerator_cpu_ptr, GENERATIONS,
                                       upper_bounds_ptr, lower_bounds_ptr);
  solver.setup_operators(&cpru, &csu, &cpu);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

TEST_P(CPUFP8E5M2Test, PSO_FP8_E5M2) {
  CanonicalParticleRecordUpdate<__nv_fp8_e5m2> cpru;
  CanonicalSpeedUpdate<__nv_fp8_e5m2> csu;
  CanonicalPositionUpdate<__nv_fp8_e5m2> cpu;

  pso_solver_cpu<__nv_fp8_e5m2> solver(population_cpu_ptr, evaluator_cpu_ptr,
                                       prngenerator_cpu_ptr, GENERATIONS,
                                       upper_bounds_ptr, lower_bounds_ptr);
  solver.setup_operators(&cpru, &csu, &cpu);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

INSTANTIATE_TEST_CASE_P(
    CPUHalfPrecisionSuite, CPUHalfPrecisionTest,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    CPUBFloat16Suite, CPUBFloat16Test,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    CPUFP8E4M3Suite, CPUFP8E4M3Test,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    CPUFP8E5M2Suite, CPUFP8E5M2Test,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

int main(int argc, char **argv) {
  BenchmarkFunctions = {1, 9};
  IslesCombinations = {8, 16, 32, 64};
  AgentsCombinations = {256, 512};
  DimensionsCombinations = {256, 512, 1024, 2048};

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
