#include "gtest/gtest.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
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

// =================================================================================================
// PRNG Allocator Optimization and Robustness Test Suite
// =================================================================================================

template <typename TFloat>
void test_prng_robustness() {
  uint32_t num_engines = 256;
  prngenerator_cuda<TFloat> generator(num_engines);

  // ---------------------------------------------------------------------------------------------
  // Runtime Random Source Seed Selection & Derived Seed Generation
  // ---------------------------------------------------------------------------------------------
  uint64_t source_seed;
  const char* env_seed = std::getenv("LOCUSTA_PRNG_SEED");
  if (env_seed != nullptr) {
    source_seed = std::stoull(env_seed);
    std::cout << "[PRNG Test] Using environment source_seed: " << source_seed << std::endl;
  } else {
    std::random_device rd;
    source_seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    std::cout << "[PRNG Test] Runtime source_seed: " << source_seed << std::endl;
  }

  std::mt19937_64 rng(source_seed);
  std::vector<uint64_t> seeds = {rng(), rng(), rng()};

  // ---------------------------------------------------------------------------------------------
  // Buffer Resizing, Unaligned Batch Sizes, and Multi-Seed Sweeps
  // ---------------------------------------------------------------------------------------------
  std::vector<uint32_t> test_sizes = {1, 1007, 65537, 3333, 17, 100000};

  for (uint64_t seed : seeds) {
    std::cout << "[PRNG Test] Testing Seed: " << seed << " (source_seed: " << source_seed << ")" << std::endl;
    SCOPED_TRACE("Testing PRNG with seed: " + std::to_string(seed) + " (source_seed: " + std::to_string(source_seed) + ")");

    generator._initialize_engines(seed);

    for (uint32_t size : test_sizes) {
      TFloat* d_out;
      CudaSafeCall(cudaMalloc((void**)&d_out, size * sizeof(TFloat)));
      generator._generate(size, d_out);

      // Range check sample
      uint32_t sample_n = std::min(size, 100u);
      std::vector<TFloat> h_sample(sample_n);
      CudaSafeCall(cudaMemcpy(h_sample.data(), d_out, sample_n * sizeof(TFloat), cudaMemcpyDeviceToHost));

      for (uint32_t i = 0; i < sample_n; ++i) {
        float val = static_cast<float>(h_sample[i]);
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
      }
      CudaSafeCall(cudaFree(d_out));
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Large-Sample Statistical Moments, Serial Independence, and Chi-Square Distribution Analysis
  // ---------------------------------------------------------------------------------------------
  uint64_t stat_seed = rng();
  std::cout << "[PRNG Test] Testing Statistical Distribution with Seed: " << stat_seed << std::endl;
  SCOPED_TRACE("Testing Statistical Distribution with seed: " + std::to_string(stat_seed));

  generator._initialize_engines(stat_seed);
  const uint32_t N = 100000;
  TFloat* d_out;
  CudaSafeCall(cudaMalloc((void**)&d_out, N * sizeof(TFloat)));
  generator._generate(N, d_out);

  std::vector<TFloat> h_data(N);
  CudaSafeCall(cudaMemcpy(h_data.data(), d_out, N * sizeof(TFloat), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(d_out));

  double sum = 0.0;
  const uint32_t num_bins = 10;
  std::vector<uint32_t> bin_counts(num_bins, 0);

  for (uint32_t i = 0; i < N; ++i) {
    float val = static_cast<float>(h_data[i]);
    sum += val;

    int bin = static_cast<int>(val * num_bins);
    if (bin >= static_cast<int>(num_bins)) bin = num_bins - 1;
    if (bin < 0) bin = 0;
    bin_counts[bin]++;
  }

  // 1. Mean check (theoretical U(0,1) mean = 0.5)
  double mean = sum / N;
  EXPECT_NEAR(mean, 0.5, 0.015);

  // 2. Variance check (theoretical U(0,1) variance = 1/12 ≈ 0.083333)
  double sq_diff_sum = 0.0;
  double lag1_cov_sum = 0.0;
  for (uint32_t i = 0; i < N; ++i) {
    float val = static_cast<float>(h_data[i]);
    double diff = val - mean;
    sq_diff_sum += diff * diff;

    if (i < N - 1) {
      float next_val = static_cast<float>(h_data[i + 1]);
      lag1_cov_sum += diff * (next_val - mean);
    }
  }
  double variance = sq_diff_sum / N;
  EXPECT_NEAR(variance, 1.0 / 12.0, 0.01);

  // 3. Lag-1 Serial Autocorrelation / Covariance Check
  double lag1_covariance = lag1_cov_sum / (N - 1);
  EXPECT_NEAR(lag1_covariance, 0.0, 0.005);

  // 4. Chi-Square Goodness-of-Fit Test
  double expected_per_bin = static_cast<double>(N) / num_bins;
  double chi_square = 0.0;
  for (uint32_t b = 0; b < num_bins; ++b) {
    double dev = bin_counts[b] - expected_per_bin;
    chi_square += (dev * dev) / expected_per_bin;
  }

  if (sizeof(TFloat) >= 2) {
    // 16-bit (half, bfloat16), 32-bit (float), 64-bit (double)
    EXPECT_LT(chi_square, 50.0);
  } else {
    // 8-bit FP8 (E4M3, E5M2) quantization binning density check
    EXPECT_LT(chi_square, 7000.0);
  }
}

TEST(PRNGRobustnessTest, Precision_Float) { test_prng_robustness<float>(); }
TEST(PRNGRobustnessTest, Precision_Double) { test_prng_robustness<double>(); }
TEST(PRNGRobustnessTest, Precision_Half) { test_prng_robustness<__half>(); }
TEST(PRNGRobustnessTest, Precision_BFloat16) { test_prng_robustness<__nv_bfloat16>(); }
TEST(PRNGRobustnessTest, Precision_FP8_E4M3) { test_prng_robustness<__nv_fp8_e4m3>(); }
TEST(PRNGRobustnessTest, Precision_FP8_E5M2) { test_prng_robustness<__nv_fp8_e5m2>(); }

int main(int argc, char **argv) {
  BenchmarkFunctions = {1, 9};
  IslesCombinations = {8, 16, 32, 64};
  AgentsCombinations = {256, 512};
  DimensionsCombinations = {256, 512, 1024, 2048};

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
