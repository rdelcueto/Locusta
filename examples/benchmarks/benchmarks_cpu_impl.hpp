#ifndef _BENCHMARKS_CPU_IMPL_H_
#define _BENCHMARKS_CPU_IMPL_H_

namespace locusta {

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
inline TFloat
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
inline TFloat
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
inline TFloat
asy(const TFloat x, const TFloat beta, const uint32_t i, const uint32_t k)
{
  float fx = static_cast<float>(x);
  float fbeta = static_cast<float>(beta);
  if (fx > 0.0f) {
    float res = std::pow(fx, 1.0f + fbeta * i / (k - 1) * std::sqrt(fx));
    return static_cast<TFloat>(res);
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
inline TFloat
osz(const TFloat x, const uint32_t i, const uint32_t k)
{
  const float c3 = 0.049f;
  float fx = static_cast<float>(x);

  if (i == 0 || i == (k - 1)) {

    if (fx == 0.0f) {
      return static_cast<TFloat>(0.0f);
    } else {
      const float xx = std::log(std::abs(fx));
      if (fx > 0.0f) {
        const float c1 = 10.0f;
        const float c2 = 7.9f;

        return static_cast<TFloat>(std::exp(xx + c3 * (std::sin(c1 * xx) + std::sin(c2 * xx))));
      } else {
        const float c1 = 5.5f;
        const float c2 = 3.1f;

        return static_cast<TFloat>(-std::exp(xx + c3 * (std::sin(c1 * xx) + std::sin(c2 * xx))));
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
inline TFloat
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

/**
 * @brief High Conditioned Elliptic function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
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
    reduction_sum += std::pow(c1, c2 * k / (DIMENSIONS - 1)) * osz_x * osz_x;
  }
  return static_cast<TFloat>(reduction_sum);
}
/**
 * @brief Bent Cigar function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
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
      reduction_sum += std::pow(c1, c2) * asy_x * asy_x;
    }
  }

  return static_cast<TFloat>(reduction_sum);
}
/**
 * @brief Discus function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
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
      reduction_sum += std::pow(c1, c2) * osz_x * osz_x;
    } else {
      reduction_sum += osz_x * osz_x;
    }
  }
  return static_cast<TFloat>(reduction_sum);
}

/**
 * @brief Different Powers function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
diff_powers(const uint32_t DIMENSIONS,
            const uint32_t DIMENSION_OFFSET,
            const TFloat* evaluation_vector)
{
  float reduction_sum = 0.0f;

  for (uint32_t k = 0; k < DIMENSIONS; ++k) {
    const float x = static_cast<float>(evaluation_vector[k * DIMENSION_OFFSET]);
    reduction_sum += std::pow(std::abs(x), 2.0f + 4.0f * k / (DIMENSIONS - 1));
  }

  reduction_sum = std::sqrt(reduction_sum);
  return static_cast<TFloat>(reduction_sum);
}

/**
 * @brief Rosenbrock’s function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
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

    x = x_1; // Update x value.
  }
  return static_cast<TFloat>(reduction_sum);
}

/**
 * @brief Schaffers F7 function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
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
  float y = asy_x * std::pow(c1, c2 * 0 / (DIMENSIONS - 1) / c3);

  float reduction_sum = 0.0f;
  for (uint32_t k = 0; k < DIMENSIONS - 1; ++k) {
    const float x_1 = static_cast<float>(evaluation_vector[(k + 1) * DIMENSION_OFFSET]);
    const float asy_x_1 = static_cast<float>(asy<TFloat>(static_cast<TFloat>(x_1), static_cast<TFloat>(beta), (k + 1), DIMENSIONS));
    const float y_1 = asy_x_1 * std::pow(c1, c2 * k / (DIMENSIONS - 1) / c3);
    const float z = std::sqrt(y * y + y_1 * y_1);

    const float tmp = std::sin(50.0f * std::pow(z, c5));
    reduction_sum += std::sqrt(z) + std::sqrt(z) * tmp * tmp;

    y = y_1;
  }
  return static_cast<TFloat>(reduction_sum);
}

/**
 * @brief Ackley’s function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
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
    const float y = asy_x * std::pow(c1, c2 * k / (DIMENSIONS - 1) / c3);

    reduction_sum_a += y * y;
    reduction_sum_b += std::cos(c3 * PI * y);
  }

  reduction_sum_a = -c4 * std::sqrt(reduction_sum_a / DIMENSIONS);
  reduction_sum_b /= DIMENSIONS;

  return static_cast<TFloat>(E - c5 * std::exp(reduction_sum_a) - std::exp(reduction_sum_b) + c5);
}

/**
 * @brief Weierstrass function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
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
    const float y = asy_x * std::pow(c1, c2 * k / (DIMENSIONS - 1) / c3);

    float pow_a = 1.0f;
    float pow_b = 1.0f;

    for (uint32_t it = 0; it <= it_max; ++it) {
      reduction_sum_b += pow_a * std::cos(c3 * PI * pow_b * (y + c4));
      reduction_sum_c += pow_a * std::cos(c3 * PI * pow_b * c4);
      pow_a *= a;
      pow_b *= b;
    }

    reduction_sum_a += reduction_sum_b - DIMENSIONS * reduction_sum_c;
  }
  return static_cast<TFloat>(reduction_sum_a);
}

/**
 * @brief Griewank’s function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
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
    const float y = x * std::pow(c1, c2 * k / (DIMENSIONS - 1) / c3);
    reduction_sum += y * y;
    reduction_prod *= std::cos(y / std::sqrt(c2 + k));
  }
  return static_cast<TFloat>(c4 + reduction_sum / c5 - reduction_prod);
}

/**
 * @brief Rastrigin’s function.
 *
 * @param DIMENSIONS Number of dimensions.
 * @param DIMENSION_OFFSET Offset for the dimension index.
 * @param evaluation_vector Vector of values to evaluate.
 * @return Result of the evaluation.
 */
template<typename TFloat>
inline TFloat
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
    const float y = std::pow(alpha, c1 * k / (DIMENSIONS - 1) / 2.0f);
    reduction_sum += (y * y - c3 * std::cos(c2 * PI * y)) + c3;
  }

  return static_cast<TFloat>(reduction_sum);
}

/**
 * @brief Dispatch function for benchmark functions.
 *
 * This function dispatches the evaluation of a benchmark function to the
 * appropriate implementation.
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
 * @param local_generator Pseudo-random number generator.
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
                   prngenerator_cpu<TFloat>* local_generator)
{

  const uint32_t REPETITIONS = 1e0;
  TFloat (*benchmark_function)(const uint32_t, const uint32_t, const TFloat*);

  switch (FUNC_ID) {

    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_ELLIPS:
      benchmark_function = ellips;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_BENT_CIGAR:
      benchmark_function = bent_cigar;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_DISCUS:
      benchmark_function = discus;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::DIFF_POWERS:
      benchmark_function = diff_powers;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_ROSENBROCK:
      benchmark_function = rosenbrock;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_SCHAFFER:
      benchmark_function = schaffer;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_ACKLEY:
      benchmark_function = ackley;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_WEIERSTRASS:
      benchmark_function = weierstrass;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_GRIEWANK:
      benchmark_function = griewank;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::RASTRIGIN:
      benchmark_function = rastrigin;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_RASTRIGIN:
      benchmark_function = rastrigin;
      break;
    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::SPHERE:
    default:
      benchmark_function = sphere;
      break;
  }

#pragma omp parallel for collapse(2)
  for (uint32_t i = 0; i < ISLES; ++i) {
    for (uint32_t j = 0; j < AGENTS; ++j) {
      const uint32_t isle = i;
      const uint32_t agent = j;
      const uint32_t OFFSET = 1;

      TFloat result = static_cast<TFloat>(0.0f);
      const TFloat* genome =
        evaluation_data + i * AGENTS * DIMENSIONS + j * DIMENSIONS;

      for (uint32_t r = 0; r < REPETITIONS; ++r) {
        result = benchmark_function(DIMENSIONS, OFFSET, genome);
      }

      result = static_cast<TFloat>(static_cast<float>(result) + static_cast<float>(EVALUATION_BIAS));

      float fres = static_cast<float>(result);
      evaluation_results[isle * AGENTS + agent] =
        F_NEGATE_EVALUATION ? static_cast<TFloat>(-fres) : result;
    }
  }
}
}

#endif /* _BENCHMARKS_CPU_IMPL */
