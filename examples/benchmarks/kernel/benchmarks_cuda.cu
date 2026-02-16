#include <math_functions.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "../benchmarks_cuda.hpp"
#include "cuda_common/cuda_helpers.h"

namespace locusta {

/// GPU Kernels Shared Memory Pointer.
extern __shared__ int evaluator_shared_memory[];
const uint32_t REPETITIONS = 1e0;

/**
 * @brief Scale a value.
 *
 * This function scales a value by a given factor.
 *
 * @param x Value to scale.
 * @param scale Scaling factor.
 * @return Scaled value.
 */
template<typename TFloat>
__device__ inline TFloat
scale(const TFloat x, const TFloat scale)
{
  return static_cast<TFloat>(static_cast<float>(x) * static_cast<float>(scale));
}
/**
 * @brief Shift a value.
 *
 * This function shifts a value by a given offset.
 *
 * @param x Value to shift.
 * @param o Offset value.
 * @return Shifted value.
 */
template<typename TFloat>
__device__ inline TFloat
shift(const TFloat x, const TFloat o)
{
  return static_cast<TFloat>(static_cast<float>(x) - static_cast<float>(o));
}

/**
 * @brief Apply the asy transformation.
 *
 * This function applies the asy transformation to a value.
 *
 * @param x Value to transform.
 * @param beta Beta parameter.
 * @param i Index of the dimension.
 * @param k Total number of dimensions.
 * @return Transformed value.
 */
template<typename TFloat>
__device__ inline TFloat
asy(const TFloat x, const TFloat beta, const uint32_t i, const uint32_t k)
{
  float fx = static_cast<float>(x);
  float fbeta = static_cast<float>(beta);
  if (fx > 0.0f) {
    return static_cast<TFloat>(powf(fx, 1.0f + fbeta * i / (k - 1) * sqrtf(fx)));
  } else {
    return x;
  }
}

/**
 * @brief Apply the osz transformation.
 *
 * This function applies the osz transformation to a value.
 *
 * @param x Value to transform.
 * @param i Index of the dimension.
 * @param k Total number of dimensions.
 * @return Transformed value.
 */
template<typename TFloat>
__device__ inline TFloat
osz(const TFloat x, const uint32_t i, const uint32_t k)
{
  const float c3 = 0.049f;
  float fx = static_cast<float>(x);

  if (i == 0 || i == (k - 1)) {

    if (fx == 0.0f) {
      return static_cast<TFloat>(0.0);
    } else {
      const float xx = logf(fabsf(fx));
      if (fx > 0.0f) {
        const float c1 = 10.0f;
        const float c2 = 7.9f;

        return static_cast<TFloat>(expf(xx + c3 * (sinf(c1 * xx) + sinf(c2 * xx))));
      } else {
        const float c1 = 5.5f;
        const float c2 = 3.1f;

        return static_cast<TFloat>(-expf(xx + c3 * (sinf(c1 * xx) + sinf(c2 * xx))));
      }
    }

  } else {
    return x;
  }
}
/**
 * @brief Sphere function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
__device__ inline TFloat
sphere(const uint32_t DIMENSIONS,
       const uint32_t DIMENSION_OFFSET,
       const TFloat* evaluation_vector)
{
  float reduction_sum = 0.0f;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const float x = static_cast<float>(evaluation_vector[k * DIMENSION_OFFSET]);
    reduction_sum += x * x;
  }
  return static_cast<TFloat>(reduction_sum);
}

template<typename TFloat>
__device__ inline TFloat
ellips(const uint32_t DIMENSIONS,
       const uint32_t DIMENSION_OFFSET,
       const TFloat* evaluation_vector)
{
  const float c1 = 10.0f;
  const float c2 = 6.0f;

  float reduction_sum = 0.0f;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const float x = static_cast<float>(evaluation_vector[k * DIMENSION_OFFSET]);
    const float osz_x = static_cast<float>(osz<TFloat>(static_cast<TFloat>(x), k, DIMENSIONS));
    reduction_sum += powf(c1, c2 * k / (DIMENSIONS - 1)) * osz_x * osz_x;
  }
  return static_cast<TFloat>(reduction_sum);
}

template<typename TFloat>
__device__ inline TFloat
bent_cigar(const uint32_t DIMENSIONS,
           const uint32_t DIMENSION_OFFSET,
           const TFloat* evaluation_vector)
{
  const float beta = 0.5f;
  const float c1 = 10.0f;
  const float c2 = 6.0f;
  float reduction_sum = 0.0f;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const float x = static_cast<float>(evaluation_vector[k * DIMENSION_OFFSET]);
    const float asy_x = static_cast<float>(asy<TFloat>(static_cast<TFloat>(x), static_cast<TFloat>(beta), k, DIMENSIONS));
    if (k == 0) {
      reduction_sum += asy_x * asy_x;
    } else {
      reduction_sum += powf(c1, c2) * asy_x * asy_x;
    }
  }

  return static_cast<TFloat>(reduction_sum);
}

template<typename TFloat>
__device__ inline TFloat
discus(const uint32_t DIMENSIONS,
       const uint32_t DIMENSION_OFFSET,
       const TFloat* evaluation_vector)
{
  const float c1 = 10.0f;
  const float c2 = 6.0f;

  float reduction_sum = 0.0f;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const float x = static_cast<float>(evaluation_vector[k * DIMENSION_OFFSET]);
    const float osz_x = static_cast<float>(osz<TFloat>(static_cast<TFloat>(x), k, DIMENSIONS));
    if (k == 0) {
      reduction_sum += powf(c1, c2) * osz_x * osz_x;
    } else {
      reduction_sum += osz_x * osz_x;
    }
  }
  return static_cast<TFloat>(reduction_sum);
}

template<typename TFloat>
__device__ inline TFloat
diff_powers(const uint32_t DIMENSIONS,
            const uint32_t DIMENSION_OFFSET,
            const TFloat* evaluation_vector)
{
  float reduction_sum = 0.0f;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const float x = static_cast<float>(evaluation_vector[k * DIMENSION_OFFSET]);
    reduction_sum += powf(fabsf(x), 2.0f + 4.0f * k / (DIMENSIONS - 1));
  }

  reduction_sum = sqrtf(reduction_sum);
  return static_cast<TFloat>(reduction_sum);
}

template<typename TFloat>
__device__ inline TFloat
rosenbrock(const uint32_t DIMENSIONS,
           const uint32_t DIMENSION_OFFSET,
           const TFloat* evaluation_vector)
{
  float reduction_sum = 0.0f;
  float x = static_cast<float>(evaluation_vector[0]) + 1.0f;

  for (uint32_t k = 0; k < DIMENSIONS - 1; ++k) {
    const float x_1 = static_cast<float>(evaluation_vector[(k + 1) * DIMENSION_OFFSET]) + 1.0f;
    const float y = x * x - x_1;
    const float z = x - 1.0f;
    reduction_sum += 100.0f * y * y + z * z;

    x = x_1;
  }
  return static_cast<TFloat>(reduction_sum);
}

template<typename TFloat>
__device__ inline TFloat
schaffer(const uint32_t DIMENSIONS,
         const uint32_t DIMENSION_OFFSET,
         const TFloat* evaluation_vector)
{
  const float beta = 0.5f;
  const float c1 = 10.0f;
  const float c2 = 1.0f;
  const float c3 = 2.0f;
  const float c5 = 0.2f;

  const float x = static_cast<float>(evaluation_vector[0]);
  const float asy_x = static_cast<float>(asy<TFloat>(static_cast<TFloat>(x), static_cast<TFloat>(beta), 0, DIMENSIONS));
  float y = asy_x * c1;

  float reduction_sum = 0.0f;
  for (uint32_t k = 0; k < DIMENSIONS - 1; ++k) {
    const float x_1 = static_cast<float>(evaluation_vector[(k + 1) * DIMENSION_OFFSET]);
    const float asy_x_1 = static_cast<float>(asy<TFloat>(static_cast<TFloat>(x_1), static_cast<TFloat>(beta), (k + 1), DIMENSIONS));
    const float y_1 = asy_x_1 * powf(c1, c2 * k / (DIMENSIONS - 1) / c3);
    const float z = sqrtf(y * y + y_1 * y_1);

    const float tmp = sinf(50.0f * powf(z, c5));
    reduction_sum += sqrtf(z) + sqrtf(z) * tmp * tmp;

    y = y_1;
  }
  return static_cast<TFloat>(reduction_sum);
}

template<typename TFloat>
__device__ inline TFloat
ackley(const uint32_t DIMENSIONS,
       const uint32_t DIMENSION_OFFSET,
       const TFloat* evaluation_vector)
{
  const float PI = 3.1415926535f;
  const float E = 2.7182818284f;
  const float beta = 0.5f;
  const float c1 = 10.0f;
  const float c2 = 1.0f;
  const float c3 = 2.0f;
  const float c4 = 0.2f;
  const float c5 = 20.0f;

  float reduction_sum_a = 0.0f;
  float reduction_sum_b = 0.0f;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const float x = static_cast<float>(evaluation_vector[k * DIMENSION_OFFSET]);
    const float asy_x = static_cast<float>(asy<TFloat>(static_cast<TFloat>(x), static_cast<TFloat>(beta), k, DIMENSIONS));
    const float y = asy_x * powf(c1, c2 * k / (DIMENSIONS - 1) / c3);

    reduction_sum_a += y * y;
    reduction_sum_b += cosf(c3 * PI * y);
  }

  reduction_sum_a = -c4 * sqrtf(reduction_sum_a / DIMENSIONS);
  reduction_sum_b /= DIMENSIONS;

  return static_cast<TFloat>(E - c5 * expf(reduction_sum_a) - expf(reduction_sum_b) + c5);
}

template<typename TFloat>
__device__ inline TFloat
weierstrass(const uint32_t DIMENSIONS,
            const uint32_t DIMENSION_OFFSET,
            const TFloat* evaluation_vector)
{
  const float PI = 3.14159f;
  const float beta = 0.5f;
  const float c1 = 10.0f;
  const float c2 = 1.0f;
  const float c3 = 2.0f;
  const float c4 = 0.5f;

  const float a = 0.5f;
  const float b = 3.0f;

  const uint32_t it_max = 20;

  float reduction_sum_a = 0.0f;
  float reduction_sum_b;
  float reduction_sum_c;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    reduction_sum_b = 0.0f;
    reduction_sum_c = 0.0f;

    const float x = static_cast<float>(evaluation_vector[k * DIMENSION_OFFSET]);
    const float asy_x = static_cast<float>(asy<TFloat>(static_cast<TFloat>(x), static_cast<TFloat>(beta), k, DIMENSIONS));
    const float y = asy_x * powf(c1, c2 * k / (DIMENSIONS - 1) / c3);

    float pow_a = 1.0f;
    float pow_b = 1.0f;

    for (uint32_t it = 0; it <= it_max; ++it) {
      reduction_sum_b += pow_a * cosf(c3 * PI * pow_b * (y + c4));
      reduction_sum_c += pow_a * cosf(c3 * PI * pow_b * c4);

      pow_a *= a;
      pow_b *= b;
    }

    reduction_sum_a += reduction_sum_b - DIMENSIONS * reduction_sum_c;
  }
  return static_cast<TFloat>(reduction_sum_a);
}

template<typename TFloat>
__device__ inline TFloat
griewank(const uint32_t DIMENSIONS,
         const uint32_t DIMENSION_OFFSET,
         const TFloat* evaluation_vector)
{
  const float c1 = 100.0f;
  const float c2 = 1.0f;
  const float c3 = 2.0f;
  const float c4 = 1.0f;
  const float c5 = 4000.0f;

  float reduction_sum = 0.0f;
  float reduction_prod = 1.0f;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const float x = static_cast<float>(evaluation_vector[k * DIMENSION_OFFSET]);
    const float y = x * powf(c1, c2 * k / (DIMENSIONS - 1) / c3);
    reduction_sum += y * y;
    reduction_prod *= cosf(y / sqrtf(c2 + k));
  }
  return static_cast<TFloat>(c4 + reduction_sum / c5 - reduction_prod);
}

template<typename TFloat>
__device__ inline TFloat
rastrigin(const uint32_t DIMENSIONS,
          const uint32_t DIMENSION_OFFSET,
          const TFloat* evaluation_vector)
{

  const float PI = 3.14159f;
  const float alpha = 10.0f;
  const float c1 = 1.0f;
  const float c2 = 2.0f;
  const float c3 = 10.0f;

  float reduction_sum = 0.0f;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const float y = powf(alpha, c1 * k / (DIMENSIONS - 1) / 2.0f);
    reduction_sum += (y * y - c3 * cosf(c2 * PI * y)) + c3;
  }

  return static_cast<TFloat>(reduction_sum);
}

/**
 * @brief CUDA kernel for evaluating benchmark functions.
 *
 * This kernel evaluates the fitness of a population of candidate solutions on
 * the GPU using a benchmark function.
 *
 * @param DIMENSIONS Number of dimensions per agent.
 * @param F_NEGATE_EVALUATION Flag indicating whether to negate the evaluation
 * result.
 * @param FUNC_ID Identifier of the benchmark function to evaluate.
 * @param EVALUATION_BIAS Bias value to add to the evaluation result.
 * @param SHIFT_ORIGIN Array of shift values to apply to the input data.
 * @param F_ROTATE Flag indicating whether to apply a rotation to the input
 * data.
 * @param ROTATION_MATRIX Matrix of rotation values to apply to the input data.
 * @param evaluation_data Input data to evaluate.
 * @param evaluation_results Output array to store the evaluation results.
 * @param local_generator Array of CUDA-based pseudo-random number generator
 * states.
 */
template<typename TFloat>
__global__ void
benchmark_kernel(const uint32_t DIMENSIONS,
                 const bool F_NEGATE_EVALUATION,
                 const uint32_t FUNC_ID,
                 const TFloat EVALUATION_BIAS,
                 const TFloat* __restrict__ SHIFT_ORIGIN,
                 const bool F_ROTATE,
                 const TFloat* __restrict__ ROTATION_MATRIX,
                 const TFloat* __restrict__ evaluation_data,
                 TFloat* __restrict__ evaluation_results,
                 curandState* __restrict__ local_generator)
{

  const uint32_t isle = blockIdx.x;
  const uint32_t agent = threadIdx.x;

  const uint32_t ISLES = gridDim.x;
  const uint32_t AGENTS = blockDim.x;

  const uint32_t genome_base = isle * AGENTS + agent;
  const uint32_t OFFSET = ISLES * AGENTS;

  const TFloat* genome = evaluation_data + genome_base;
  TFloat result = static_cast<TFloat>(0.0f);

  TFloat (*benchmark_function)(const uint32_t, const uint32_t, const TFloat*);

  switch (FUNC_ID) {
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_ELLIPS:
      benchmark_function = ellips;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_BENT_CIGAR:
      benchmark_function = bent_cigar;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_DISCUS:
      benchmark_function = discus;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::DIFF_POWERS:
      benchmark_function = diff_powers;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_ROSENBROCK:
      benchmark_function = rosenbrock;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_SCHAFFER:
      benchmark_function = schaffer;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_ACKLEY:
      benchmark_function = ackley;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_WEIERSTRASS:
      benchmark_function = weierstrass;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_GRIEWANK:
      benchmark_function = griewank;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::RASTRIGIN:
      benchmark_function = rastrigin;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_RASTRIGIN:
      benchmark_function = rastrigin;
      break;
    case BenchmarkCudaFunctor<TFloat>::FUNCTION_IDENTIFIERS::SPHERE:
    default:
      benchmark_function = sphere;
      break;
  }

  for (uint32_t r = 0; r < REPETITIONS; ++r) {
    result = benchmark_function(DIMENSIONS, OFFSET, genome);
  }

  result = static_cast<TFloat>(static_cast<float>(result) + static_cast<float>(EVALUATION_BIAS));

  const uint32_t fitness_idx = isle * AGENTS + agent;
  float fres = static_cast<float>(result);
  evaluation_results[fitness_idx] = F_NEGATE_EVALUATION ? static_cast<TFloat>(-fres) : result;
}

/**
 * @brief Dispatch function for benchmark functions on the GPU.
 *
 * This function dispatches the evaluation of a benchmark function to the
 * appropriate CUDA implementation.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param F_NEGATE_EVALUATION Flag indicating whether to negate the evaluation
 * result.
 * @param FUNC_ID Identifier of the benchmark function to evaluate.
 * @param EVALUATION_BIAS Bias value to add to the evaluation result.
 * @param SHIFT_ORIGIN Array of shift values to apply to the input data.
 * @param F_ROTATE Flag indicating whether to apply a rotation to the input
 * data.
 * @param ROTATION_MATRIX Matrix of rotation values to apply to the input data.
 * @param evaluation_data Input data to evaluate.
 * @param evaluation_results Output array to store the evaluation results.
 * @param local_generator CUDA-based pseudo-random number generator.
 */
template<typename TFloat>
void
benchmark_dispatch(const uint32_t ISLES,
                   const uint32_t AGENTS,
                   const uint32_t DIMENSIONS,
                   const bool F_NEGATE_EVALUATION,
                   const uint32_t FUNC_ID,
                   const TFloat EVALUATION_BIAS,
                   const TFloat* SHIFT_ORIGIN,
                   const bool F_ROTATE,
                   const TFloat* ROTATION_MATRIX,
                   const TFloat* evaluation_data,
                   TFloat* evaluation_results,
                   prngenerator_cuda<TFloat>* local_generator)
{

  curandState* device_generators =
    local_generator->get_device_generator_states();

  benchmark_kernel<<<ISLES, AGENTS>>>(DIMENSIONS,
                                      F_NEGATE_EVALUATION,
                                      FUNC_ID,
                                      EVALUATION_BIAS,
                                      SHIFT_ORIGIN,
                                      F_ROTATE,
                                      ROTATION_MATRIX,
                                      evaluation_data,
                                      evaluation_results,
                                      device_generators);

  CudaCheckError();
}

// Template Specialization (float)
template void
benchmark_dispatch<float>(const uint32_t ISLES,
                          const uint32_t AGENTS,
                          const uint32_t DIMENSIONS,
                          const bool F_NEGATE_EVALUATION,
                          const uint32_t FUNC_ID,
                          const float EVALUATION_BIAS,
                          const float* SHIFT_ORIGIN,
                          const bool F_ROTATE,
                          const float* ROTATION_MATRIX,
                          const float* evaluation_data,
                          float* evaluation_results,
                          prngenerator_cuda<float>* local_generator);

// Template Specialization (double)
template void
benchmark_dispatch<double>(const uint32_t ISLES,
                           const uint32_t AGENTS,
                           const uint32_t DIMENSIONS,
                           const bool F_NEGATE_EVALUATION,
                           const uint32_t FUNC_ID,
                           const double EVALUATION_BIAS,
                           const double* SHIFT_ORIGIN,
                           const bool F_ROTATE,
                           const double* ROTATION_MATRIX,
                           const double* evaluation_data,
                           double* evaluation_results,
                           prngenerator_cuda<double>* local_generator);

template void benchmark_dispatch<__half>(const uint32_t, const uint32_t, const uint32_t, const bool, const uint32_t, const __half, const __half*, const bool, const __half*, const __half*, __half*, prngenerator_cuda<__half>*);
template void benchmark_dispatch<__nv_bfloat16>(const uint32_t, const uint32_t, const uint32_t, const bool, const uint32_t, const __nv_bfloat16, const __nv_bfloat16*, const bool, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, prngenerator_cuda<__nv_bfloat16>*);
template void benchmark_dispatch<__nv_fp8_e4m3>(const uint32_t, const uint32_t, const uint32_t, const bool, const uint32_t, const __nv_fp8_e4m3, const __nv_fp8_e4m3*, const bool, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*, prngenerator_cuda<__nv_fp8_e4m3>*);
template void benchmark_dispatch<__nv_fp8_e5m2>(const uint32_t, const uint32_t, const uint32_t, const bool, const uint32_t, const __nv_fp8_e5m2, const __nv_fp8_e5m2*, const bool, const __nv_fp8_e5m2*, const __nv_fp8_e5m2*, __nv_fp8_e5m2*, prngenerator_cuda<__nv_fp8_e5m2>*);

}
