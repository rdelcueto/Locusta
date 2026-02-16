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

// CUDA Includes
#ifdef LOCUSTA_ENABLE_CUDA
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
#endif

using namespace locusta;

/**
 * @brief Vectors of benchmark configurations.
 */
std::vector<uint32_t> BenchmarkFunctions;
std::vector<uint32_t> IslesCombinations;
std::vector<uint32_t> AgentsCombinations;
std::vector<uint32_t> DimensionsCombinations;

const uint32_t GLOBAL_SEED = 314;
const uint32_t GLOBAL_EVALUATIONS = (64 * 64 * 64) * 1e2;

// =================================================================================================
// CPU Test Fixtures for Half & Quarter Precision Types
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

// Half Precision CPU Tests (__half and __nv_bfloat16)
class CPUHalfPrecisionTest : public CPULocustaPrecisionTest<__half> {};
class CPUBFloat16Test : public CPULocustaPrecisionTest<__nv_bfloat16> {};

// Quarter Precision CPU Tests (__nv_fp8_e4m3 and __nv_fp8_e5m2)
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

// =================================================================================================
// CUDA GPU Test Fixtures for Half & Quarter Precision Types
// =================================================================================================

#ifdef LOCUSTA_ENABLE_CUDA

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

// GPU Half Precision Tests (__half and __nv_bfloat16)
class GPUHalfPrecisionTest : public GPULocustaPrecisionTest<__half> {};
class GPUBFloat16Test : public GPULocustaPrecisionTest<__nv_bfloat16> {};

// GPU Quarter Precision Tests (__nv_fp8_e4m3 and __nv_fp8_e5m2)
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

#endif // LOCUSTA_ENABLE_CUDA

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
  BenchmarkFunctions = {1, 9};

  IslesCombinations = {8, 16, 32, 64};

  AgentsCombinations = {256, 512};

  DimensionsCombinations = {256, 512, 1024, 2048};

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
