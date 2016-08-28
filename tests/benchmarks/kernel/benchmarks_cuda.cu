#include <math_functions.h>

#include "../benchmarks_cuda.hpp"
#include "cuda_common/cuda_helpers.h"

namespace locusta {

/// GPU Kernels Shared Memory Pointer.
extern __shared__ int evaluator_shared_memory[];
const uint32_t REPETITIONS = 1e2;

template <typename TFloat>
__device__ inline TFloat
scale(const TFloat x, const TFloat scale)
{
  return x * scale;
}

template <typename TFloat>
__device__ inline TFloat
shift(const TFloat x, const TFloat o)
{
  return x - o;
}

template <typename TFloat>
__device__ inline TFloat
asy(const TFloat x, const TFloat beta, const uint32_t i, const uint32_t k)
{
  const TFloat ONE = 1.0;
  // const TFloat HALF = 0.5;
  if (x > 0) {
    // return pow(x, ONE + beta*i/(k-1)*pow(x, HALF));
    return pow(x, ONE + beta * i / (k - 1) * sqrt(x));
  } else {
    return x;
  }
}

template <typename TFloat>
__device__ inline TFloat
osz(const TFloat x, const uint32_t i, const uint32_t k)
{
  const TFloat c3 = 0.049;

  if (i == 0 || i == (k - 1)) {

    if (x == 0) {
      return 0;
    } else {
      const TFloat xx = log(fabs(x));
      if (x > 0) {
        const TFloat c1 = 10;
        const TFloat c2 = 7.9;

        return exp(xx + c3 * (sin(c1 * xx) + sin(c2 * xx)));
      } else {
        const TFloat c1 = 5.5;
        const TFloat c2 = 3.1;

        return -exp(xx + c3 * (sin(c1 * xx) + sin(c2 * xx)));
      }
    }

  } else {
    return x;
  }
}

template <typename TFloat>
__device__ inline TFloat
sphere(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
       const TFloat* evaluation_vector)
{
  TFloat reduction_sum = 0.0;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
    reduction_sum += x * x;
  }
  return reduction_sum;
}

template <typename TFloat>
__device__ inline TFloat
ellips(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
       const TFloat* evaluation_vector)
{
  const TFloat c1 = 10.0;
  const TFloat c2 = 6.0;

  TFloat reduction_sum = 0.0;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
    const TFloat osz_x = osz<TFloat>(x, k, DIMENSIONS);
    reduction_sum += pow(c1, c2 * k / (DIMENSIONS - 1)) * osz_x * osz_x;
  }
  return reduction_sum;
}

template <typename TFloat>
__device__ inline TFloat
bent_cigar(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
           const TFloat* evaluation_vector)
{
  const TFloat beta = 0.5;
  const TFloat c1 = 10.0;
  const TFloat c2 = 6.0;
  TFloat reduction_sum = 0;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
    const TFloat asy_x = asy<TFloat>(x, beta, k, DIMENSIONS);
    if (k == 0) {
      reduction_sum += asy_x * asy_x;
    } else {
      reduction_sum += pow(c1, c2) * asy_x * asy_x;
    }
  }

  return reduction_sum;
}

template <typename TFloat>
__device__ inline TFloat
discus(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
       const TFloat* evaluation_vector)
{
  const TFloat c1 = 10.0;
  const TFloat c2 = 6.0;

  TFloat reduction_sum = 0.0;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
    const TFloat osz_x = osz<TFloat>(x, k, DIMENSIONS);
    if (k == 0) {
      reduction_sum += pow(c1, c2) * osz_x * osz_x;
    } else {
      reduction_sum += osz_x * osz_x;
    }
  }
  return reduction_sum;
}

template <typename TFloat>
__device__ inline TFloat
diff_powers(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
            const TFloat* evaluation_vector)
{
  const TFloat ONE = 1.0;
  // const TFloat c1 = 0.5;
  TFloat reduction_sum = 0.0;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
    reduction_sum += pow(fabs(x), 2 + 4 * k / (DIMENSIONS - 1) * ONE);
  }

  // reduction_sum = pow(reduction_sum, c1);
  reduction_sum = sqrt(reduction_sum);
  return reduction_sum;
}

template <typename TFloat>
__device__ inline TFloat
rosenbrock(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
           const TFloat* evaluation_vector)
{
  TFloat reduction_sum = 0.0;
  TFloat x = evaluation_vector[0] + 1;

  for (uint32_t k = 0; k < DIMENSIONS - 1; ++k) {
    const TFloat x_1 = evaluation_vector[(k + 1) * DIMENSION_OFFSET] + 1;
    const TFloat y = x * x - x_1;
    const TFloat z = x - 1.0;
    reduction_sum += 100 * y * y + z * z;

    x = x_1; // Update x value.
  }
  return reduction_sum;
}

template <typename TFloat>
__device__ inline TFloat
schaffer(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
         const TFloat* evaluation_vector)
{
  const TFloat beta = 0.5;
  const TFloat c1 = 10.0;
  const TFloat c2 = 1.0;
  const TFloat c3 = 2.0;
  // const TFloat c4 = 0.5;
  const TFloat c5 = 0.2;

  const TFloat x = evaluation_vector[0];
  const TFloat asy_x = asy<TFloat>(x, beta, 0, DIMENSIONS);
  // TFloat y = asy_x * pow(c1, c2*0/(DIMENSIONS-1)/c3);
  TFloat y = asy_x * c1;

  TFloat reduction_sum = 0.0;
  for (uint32_t k = 0; k < DIMENSIONS - 1; ++k) {
    const TFloat x_1 = evaluation_vector[(k + 1) * DIMENSION_OFFSET];
    const TFloat asy_x_1 = asy<TFloat>(x_1, beta, (k + 1), DIMENSIONS);
    const TFloat y_1 = asy_x_1 * pow(c1, c2 * k / (DIMENSIONS - 1) / c3);
    const TFloat z = sqrt(y * y + y_1 * y_1);

    const TFloat tmp = sin(50 * pow(z, c5));
    // reduction_sum += pow(z, c4) + pow(z, c4) * tmp*tmp;
    reduction_sum += sqrt(z) + sqrt(z) * tmp * tmp;

    y = y_1;
  }
  return reduction_sum;
}

template <typename TFloat>
__device__ inline TFloat
ackley(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
       const TFloat* evaluation_vector)
{
  const TFloat PI = 3.1415926535;
  const TFloat E = 2.7182818284;
  const TFloat beta = 0.5;
  const TFloat c1 = 10.0;
  const TFloat c2 = 1.0;
  const TFloat c3 = 2.0;
  const TFloat c4 = 0.2;
  const TFloat c5 = 20.0;

  TFloat reduction_sum_a = 0.0;
  TFloat reduction_sum_b = 0.0;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
    const TFloat asy_x = asy<TFloat>(x, beta, k, DIMENSIONS);
    const TFloat y = asy_x * pow(c1, c2 * k / (DIMENSIONS - 1) / c3);

    reduction_sum_a += y * y;
    reduction_sum_b += cos(c3 * PI * y);
  }

  reduction_sum_a = -c4 * sqrt(reduction_sum_a / DIMENSIONS);
  reduction_sum_b /= DIMENSIONS;

  return E - c5 * exp(reduction_sum_a) - exp(reduction_sum_b) + c5;
}

template <typename TFloat>
__device__ inline TFloat
weierstrass(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
            const TFloat* evaluation_vector)
{
  const TFloat PI = 3.14159;
  const TFloat beta = 0.5;
  const TFloat c1 = 10.0;
  const TFloat c2 = 1.0;
  const TFloat c3 = 2.0;
  const TFloat c4 = 0.5;

  const TFloat a = 0.5;
  const TFloat b = 3.0;

  const uint32_t it_max = 20;

  TFloat reduction_sum_a = 0.0;
  TFloat reduction_sum_b;
  TFloat reduction_sum_c;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    reduction_sum_b = 0.0;
    reduction_sum_c = 0.0;

    const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
    const TFloat asy_x = asy<TFloat>(x, beta, k, DIMENSIONS);
    const TFloat y = asy_x * pow(c1, c2 * k / (DIMENSIONS - 1) / c3);

    TFloat pow_a = 1;
    TFloat pow_b = 1;

    for (uint32_t it = 0; it <= it_max; ++it) {
      reduction_sum_b += pow_a * cos(c3 * PI * pow_b * (y + c4));
      reduction_sum_c += pow_a * cos(c3 * PI * pow_b * c4);

      pow_a *= a;
      pow_b *= b;
    }

    reduction_sum_a += reduction_sum_b - DIMENSIONS * reduction_sum_c;
  }
  return reduction_sum_a;
}

template <typename TFloat>
__device__ inline TFloat
griewank(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
         const TFloat* evaluation_vector)
{
  const TFloat c1 = 100.0;
  const TFloat c2 = 1.0;
  const TFloat c3 = 2.0;
  const TFloat c4 = 1.0;
  const TFloat c5 = 4000.0;

  TFloat reduction_sum = 0.0;
  TFloat reduction_prod = 1.0;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
    const TFloat y = x * pow(c1, c2 * k / (DIMENSIONS - 1) / c3);
    reduction_sum += y * y;
    reduction_prod *= cos(y / sqrt(c2 + k));
  }
  return c4 + reduction_sum / c5 - reduction_prod;
}

template <typename TFloat>
__device__ inline TFloat
rastrigin(const uint32_t DIMENSIONS, const uint32_t DIMENSION_OFFSET,
          const TFloat* evaluation_vector)
{

  const TFloat PI = 3.14159;
  const TFloat alpha = 10.0;
  const TFloat beta = 0.2;
  const TFloat c1 = 1.0;
  const TFloat c2 = 2.0;
  const TFloat c3 = 10.0;

  TFloat reduction_sum = 0.0;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
    const TFloat osz_x = osz<TFloat>(x, k, DIMENSIONS);
    const TFloat asy_x = asy<TFloat>(osz_x, beta, k, DIMENSIONS);
    const TFloat y = pow(alpha, c1 * k / (DIMENSIONS - 1) / 2);

    reduction_sum += (y * y - c3 * cos(c2 * PI * y)) + c3;
  }

  return reduction_sum;
}

template <typename TFloat>
__global__ void
benchmark_kernel(const uint32_t DIMENSIONS, const bool F_NEGATE_EVALUATION,
                 const uint32_t FUNC_ID, const TFloat EVALUATION_BIAS,
                 const TFloat* __restrict__ SHIFT_ORIGIN, const bool F_ROTATE,
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
  TFloat result = 0;

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

  result += EVALUATION_BIAS;

  const uint32_t fitness_idx = isle * AGENTS + agent;
  evaluation_results[fitness_idx] = F_NEGATE_EVALUATION ? -result : result;
}

template <typename TFloat>
void
benchmark_dispatch(const uint32_t ISLES, const uint32_t AGENTS,
                   const uint32_t DIMENSIONS, const bool F_NEGATE_EVALUATION,
                   const uint32_t FUNC_ID, const TFloat EVALUATION_BIAS,
                   const TFloat* SHIFT_ORIGIN, const bool F_ROTATE,
                   const TFloat* ROTATION_MATRIX, const TFloat* evaluation_data,
                   TFloat* evaluation_results,
                   prngenerator_cuda<TFloat>* local_generator)
{

  curandState* device_generators =
    local_generator->get_device_generator_states();

  benchmark_kernel<<<ISLES, AGENTS>>>(DIMENSIONS, F_NEGATE_EVALUATION, FUNC_ID,
                                      EVALUATION_BIAS, SHIFT_ORIGIN, F_ROTATE,
                                      ROTATION_MATRIX, evaluation_data,
                                      evaluation_results, device_generators);

  CudaCheckError();
}

// Template Specialization (float)
template void benchmark_dispatch<float>(
  const uint32_t ISLES, const uint32_t AGENTS, const uint32_t DIMENSIONS,
  const bool F_NEGATE_EVALUATION, const uint32_t FUNC_ID,
  const float EVALUATION_BIAS, const float* SHIFT_ORIGIN, const bool F_ROTATE,
  const float* ROTATION_MATRIX, const float* evaluation_data,
  float* evaluation_results, prngenerator_cuda<float>* local_generator);

// Template Specialization (double)
template void benchmark_dispatch<double>(
  const uint32_t ISLES, const uint32_t AGENTS, const uint32_t DIMENSIONS,
  const bool F_NEGATE_EVALUATION, const uint32_t FUNC_ID,
  const double EVALUATION_BIAS, const double* SHIFT_ORIGIN, const bool F_ROTATE,
  const double* ROTATION_MATRIX, const double* evaluation_data,
  double* evaluation_results, prngenerator_cuda<double>* local_generator);
}
