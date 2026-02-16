#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <time.h>
#include <tuple>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// CUDA Includes
#include "./benchmarks/benchmarks_cuda.hpp"
#include "cuda_runtime.h"
#include "evaluator/evaluator_cuda.hpp"
#include "prngenerator/prngenerator_cuda.hpp"
#include "solvers/de/de_operators/de_std_operators_cuda_impl.hpp"
#include "solvers/de/de_solver_cuda.hpp"
#include "solvers/ga/ga_operators/ga_std_operators_cuda_impl.hpp"
#include "solvers/ga/ga_solver_cuda.hpp"
#include "solvers/pso/pso_operators/pso_std_operators_cuda_impl.hpp"
#include "solvers/pso/pso_solver_cuda.hpp"

using namespace locusta;

std::vector<uint32_t> BenchmarkFunctions;
std::vector<uint32_t> IslesCombinations;
std::vector<uint32_t> AgentsCombinations;
std::vector<uint32_t> DimensionsCombinations;

const uint32_t GLOBAL_SEED = 314;
const uint32_t GLOBAL_EVALUATIONS = (64 * 64 * 64) * 1e2;

// =================================================================================================
// Standard Single Precision CUDA Tests
// =================================================================================================

class GPULocustaTest : public testing::TestWithParam<
                           std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> {

public:
  virtual ~GPULocustaTest() {}

protected:
  virtual void SetUp() {
    GENERATIONS = GENERATIONS != 0 ? GENERATIONS : 1;

    __setup_cuda(0, 1);

    upper_bounds_ptr = new float[DIMENSIONS];
    lower_bounds_ptr = new float[DIMENSIONS];

    // Bounds definition
    std::fill(upper_bounds_ptr, upper_bounds_ptr + DIMENSIONS, 100.0f);
    std::fill(lower_bounds_ptr, lower_bounds_ptr + DIMENSIONS, -100.0f);

    evaluation_functor_cuda_ptr =
        new BenchmarkCudaFunctor<float>(BENCHMARK_FUNC_ID, DIMENSIONS);

    evaluator_cuda_ptr =
        new evaluator_cuda<float>(evaluation_functor_cuda_ptr, true,
                                  BoundMapKind::CropBounds, DIMENSIONS);

    prngenerator_cuda_ptr = new prngenerator_cuda<float>(ISLES * AGENTS);
    prngenerator_cuda_ptr->_initialize_engines(SEED);

    population_cuda_ptr =
        new population_set_cuda<float>(ISLES, AGENTS, DIMENSIONS);

    std::cout << "FuncId: " << BENCHMARK_FUNC_ID
              << ", Generations: " << GENERATIONS << ", Isles: " << ISLES
              << ", Agents: " << AGENTS << ", Dimensions: " << DIMENSIONS
              << ", CV: " << (GENERATIONS * ISLES * AGENTS * DIMENSIONS)
              << std::endl;
  }

  virtual void TearDown() {
    delete population_cuda_ptr;
    population_cuda_ptr = NULL;

    delete prngenerator_cuda_ptr;
    prngenerator_cuda_ptr = NULL;

    delete evaluator_cuda_ptr;
    evaluator_cuda_ptr = NULL;

    delete evaluation_functor_cuda_ptr;
    evaluation_functor_cuda_ptr = NULL;

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

  uint32_t const GPU_FACTOR = 1e0;
  uint32_t const FUNC_WEIGHT = BENCHMARK_FUNC_ID == 1 ? 1e1 : 2e0;
  uint64_t GENERATIONS =
      (FUNC_WEIGHT * GLOBAL_EVALUATIONS) / (ISLES * AGENTS * DIMENSIONS);

  float *upper_bounds_ptr;
  float *lower_bounds_ptr;

  prngenerator_cuda<float> *prngenerator_cuda_ptr;
  EvaluationCudaFunctor<float> *evaluation_functor_cuda_ptr;
  evaluator_cuda<float> *evaluator_cuda_ptr;
  population_set_cuda<float> *population_cuda_ptr;
};

class GPUParticleSwarmTest : public GPULocustaTest {
public:
  virtual ~GPUParticleSwarmTest() {}

protected:
  virtual void SetUp() {
    GPULocustaTest::SetUp();
    pso_solver_cuda_ptr = new pso_solver_cuda<float>(
        population_cuda_ptr, evaluator_cuda_ptr, prngenerator_cuda_ptr,
        GENERATIONS, upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown() {
    delete pso_solver_cuda_ptr;
    pso_solver_cuda_ptr = NULL;
    GPULocustaTest::TearDown();
  }

public:
  pso_solver_cuda<float> *pso_solver_cuda_ptr;
};

class GPUGeneticAlgorithmTest : public GPULocustaTest {
public:
  virtual ~GPUGeneticAlgorithmTest() {}

protected:
  virtual void SetUp() {
    GPULocustaTest::SetUp();
    ga_solver_cuda_ptr = new ga_solver_cuda<float>(
        population_cuda_ptr, evaluator_cuda_ptr, prngenerator_cuda_ptr,
        GENERATIONS, upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown() {
    delete ga_solver_cuda_ptr;
    ga_solver_cuda_ptr = NULL;
    GPULocustaTest::TearDown();
  }

public:
  ga_solver_cuda<float> *ga_solver_cuda_ptr;
};

class GPUDifferentialEvolutionTest : public GPULocustaTest {
public:
  virtual ~GPUDifferentialEvolutionTest() {}

protected:
  virtual void SetUp() {
    GPULocustaTest::SetUp();
    de_solver_cuda_ptr = new de_solver_cuda<float>(
        population_cuda_ptr, evaluator_cuda_ptr, prngenerator_cuda_ptr,
        GENERATIONS, upper_bounds_ptr, lower_bounds_ptr);
  }

  virtual void TearDown() {
    delete de_solver_cuda_ptr;
    de_solver_cuda_ptr = NULL;
    GPULocustaTest::TearDown();
  }

public:
  de_solver_cuda<float> *de_solver_cuda_ptr;
};

TEST_P(GPUParticleSwarmTest, BenchmarkCuda) {
  CanonicalParticleRecordUpdateCuda<float> cpru;
  CanonicalSpeedUpdateCuda<float> csu;
  CanonicalPositionUpdateCuda<float> cpu;

  pso_solver_cuda_ptr->setup_operators(&cpru, &csu, &cpu);
  pso_solver_cuda_ptr->setup_solver();
  pso_solver_cuda_ptr->run();
  pso_solver_cuda_ptr->teardown_solver();
}

TEST_P(GPUGeneticAlgorithmTest, BenchmarkCuda) {
  WholeCrossoverCuda<float> wc;
  TournamentSelectionCuda<float> ts;
  ga_solver_cuda_ptr->setup_operators(&wc, &ts);

  ga_solver_cuda_ptr->setup_solver();
  ga_solver_cuda_ptr->run();
  ga_solver_cuda_ptr->teardown_solver();
}

TEST_P(GPUDifferentialEvolutionTest, BenchmarkCuda) {
  DeWholeCrossoverCuda<float> dwc;
  DeRandomSelectionCuda<float> drs;
  de_solver_cuda_ptr->setup_operators(&dwc, &drs);

  de_solver_cuda_ptr->setup_solver();
  de_solver_cuda_ptr->run();
  de_solver_cuda_ptr->teardown_solver();
}

INSTANTIATE_TEST_CASE_P(
    GPUParticleSwarmTestSuite, GPUParticleSwarmTest,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    GPUGeneticAlgorithmTestSuite, GPUGeneticAlgorithmTest,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    GPUDifferentialEvolutionTestSuite, GPUDifferentialEvolutionTest,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

// =================================================================================================
// Half & Quarter Precision CUDA Tests
// =================================================================================================

template <typename TFloat>
class GPULocustaPrecisionTest
    : public testing::TestWithParam<
          std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> {
public:
  virtual ~GPULocustaPrecisionTest() {}

protected:
  virtual void SetUp() {
    GENERATIONS = GENERATIONS != 0 ? GENERATIONS : 1;

    __setup_cuda(0, 1);

    upper_bounds_ptr = new TFloat[DIMENSIONS];
    lower_bounds_ptr = new TFloat[DIMENSIONS];

    for (uint32_t i = 0; i < DIMENSIONS; ++i) {
      upper_bounds_ptr[i] = static_cast<TFloat>(100.0f);
      lower_bounds_ptr[i] = static_cast<TFloat>(-100.0f);
    }

    evaluation_functor_cuda_ptr =
        new BenchmarkCudaFunctor<TFloat>(BENCHMARK_FUNC_ID, DIMENSIONS);

    evaluator_cuda_ptr =
        new evaluator_cuda<TFloat>(evaluation_functor_cuda_ptr, true,
                                   BoundMapKind::CropBounds, DIMENSIONS);

    prngenerator_cuda_ptr = new prngenerator_cuda<TFloat>(ISLES * AGENTS);
    prngenerator_cuda_ptr->_initialize_engines(SEED);

    population_cuda_ptr =
        new population_set_cuda<TFloat>(ISLES, AGENTS, DIMENSIONS);

    std::cout << "[GPU Low-Precision Test] Type size: " << sizeof(TFloat) << "B"
              << ", FuncId: " << BENCHMARK_FUNC_ID << ", Isles: " << ISLES
              << ", Agents: " << AGENTS << ", Dimensions: " << DIMENSIONS
              << std::endl;
  }

  virtual void TearDown() {
    delete population_cuda_ptr;
    population_cuda_ptr = NULL;

    delete prngenerator_cuda_ptr;
    prngenerator_cuda_ptr = NULL;

    delete evaluator_cuda_ptr;
    evaluator_cuda_ptr = NULL;

    delete evaluation_functor_cuda_ptr;
    evaluation_functor_cuda_ptr = NULL;

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

  prngenerator_cuda<TFloat> *prngenerator_cuda_ptr;
  EvaluationCudaFunctor<TFloat> *evaluation_functor_cuda_ptr;
  evaluator_cuda<TFloat> *evaluator_cuda_ptr;
  population_set_cuda<TFloat> *population_cuda_ptr;
};

class GPUHalfPrecisionTest : public GPULocustaPrecisionTest<__half> {};
class GPUBFloat16Test : public GPULocustaPrecisionTest<__nv_bfloat16> {};
class GPUFP8E4M3Test : public GPULocustaPrecisionTest<__nv_fp8_e4m3> {};
class GPUFP8E5M2Test : public GPULocustaPrecisionTest<__nv_fp8_e5m2> {};

TEST_P(GPUHalfPrecisionTest, PSO_Cuda_Half) {
  CanonicalParticleRecordUpdateCuda<__half> cpru;
  CanonicalSpeedUpdateCuda<__half> csu;
  CanonicalPositionUpdateCuda<__half> cpu;

  pso_solver_cuda<__half> solver(population_cuda_ptr, evaluator_cuda_ptr,
                                 prngenerator_cuda_ptr, GENERATIONS,
                                 upper_bounds_ptr, lower_bounds_ptr);

  solver.setup_operators(&cpru, &csu, &cpu);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

TEST_P(GPUHalfPrecisionTest, GA_Cuda_Half) {
  WholeCrossoverCuda<__half> wc;
  TournamentSelectionCuda<__half> ts;

  ga_solver_cuda<__half> solver(population_cuda_ptr, evaluator_cuda_ptr,
                                prngenerator_cuda_ptr, GENERATIONS,
                                upper_bounds_ptr, lower_bounds_ptr);

  solver.setup_operators(&wc, &ts);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

TEST_P(GPUHalfPrecisionTest, DE_Cuda_Half) {
  DeWholeCrossoverCuda<__half> dwc;
  DeRandomSelectionCuda<__half> drs;

  de_solver_cuda<__half> solver(population_cuda_ptr, evaluator_cuda_ptr,
                                prngenerator_cuda_ptr, GENERATIONS,
                                upper_bounds_ptr, lower_bounds_ptr);

  solver.setup_operators(&dwc, &drs);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

TEST_P(GPUBFloat16Test, PSO_Cuda_BFloat16) {
  CanonicalParticleRecordUpdateCuda<__nv_bfloat16> cpru;
  CanonicalSpeedUpdateCuda<__nv_bfloat16> csu;
  CanonicalPositionUpdateCuda<__nv_bfloat16> cpu;

  pso_solver_cuda<__nv_bfloat16> solver(population_cuda_ptr, evaluator_cuda_ptr,
                                        prngenerator_cuda_ptr, GENERATIONS,
                                        upper_bounds_ptr, lower_bounds_ptr);

  solver.setup_operators(&cpru, &csu, &cpu);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

TEST_P(GPUFP8E4M3Test, PSO_Cuda_FP8_E4M3) {
  CanonicalParticleRecordUpdateCuda<__nv_fp8_e4m3> cpru;
  CanonicalSpeedUpdateCuda<__nv_fp8_e4m3> csu;
  CanonicalPositionUpdateCuda<__nv_fp8_e4m3> cpu;

  pso_solver_cuda<__nv_fp8_e4m3> solver(population_cuda_ptr, evaluator_cuda_ptr,
                                        prngenerator_cuda_ptr, GENERATIONS,
                                        upper_bounds_ptr, lower_bounds_ptr);

  solver.setup_operators(&cpru, &csu, &cpu);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

TEST_P(GPUFP8E5M2Test, PSO_Cuda_FP8_E5M2) {
  CanonicalParticleRecordUpdateCuda<__nv_fp8_e5m2> cpru;
  CanonicalSpeedUpdateCuda<__nv_fp8_e5m2> csu;
  CanonicalPositionUpdateCuda<__nv_fp8_e5m2> cpu;

  pso_solver_cuda<__nv_fp8_e5m2> solver(population_cuda_ptr, evaluator_cuda_ptr,
                                        prngenerator_cuda_ptr, GENERATIONS,
                                        upper_bounds_ptr, lower_bounds_ptr);

  solver.setup_operators(&cpru, &csu, &cpu);
  solver.setup_solver();
  solver.run();
  solver.teardown_solver();
}

INSTANTIATE_TEST_CASE_P(
    GPUHalfPrecisionSuite, GPUHalfPrecisionTest,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    GPUBFloat16Suite, GPUBFloat16Test,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    GPUFP8E4M3Suite, GPUFP8E4M3Test,
    ::testing::Combine(::testing::ValuesIn(BenchmarkFunctions),
                       ::testing::ValuesIn(IslesCombinations),
                       ::testing::ValuesIn(AgentsCombinations),
                       ::testing::ValuesIn(DimensionsCombinations)));

INSTANTIATE_TEST_CASE_P(
    GPUFP8E5M2Suite, GPUFP8E5M2Test,
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
