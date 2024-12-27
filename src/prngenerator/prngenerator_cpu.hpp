#ifndef _PRNGENERATOR_CPU_H_
#define _PRNGENERATOR_CPU_H_

#include <chrono>
#include <omp.h>
#include <random>

#include "prngenerator.hpp"

namespace locusta {

/**
 * @brief CPU implementation of the pseudo-random number generator.
 *
 * This class implements the prngenerator interface for the CPU architecture,
 * using the Mersenne Twister engine.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct prngenerator_cpu : prngenerator<TFloat>
{

  typedef std::mt19937 mersenne_twister;
  typedef std::uniform_real_distribution<TFloat> uni_real_distribution;

  /**
   * @brief Construct a new prngenerator_cpu object with a single engine.
   */
  prngenerator_cpu();

  /**
   * @brief Construct a new prngenerator_cpu object.
   *
   * @param num_engines Number of PRNG engines to use.
   */
  prngenerator_cpu(uint32_t num_engines);

  /**
   * @brief Destroy the prngenerator_cpu object.
   */
  virtual ~prngenerator_cpu();

  /**
   * @brief Initialize the state of the PRNG engines.
   *
   * @param seed Seed value for the PRNG engines.
   */
  virtual void _initialize_engines(uint64_t seed);

  /**
   * @brief Generate pseudo-random numbers.
   *
   * This method generates \p n pseudo-random numbers and stores them in the \p
   * output array.
   *
   * @param n Number of pseudo-random numbers to generate.
   * @param output Output array to store the generated pseudo-random numbers.
   */
  virtual void _generate(uint32_t n, TFloat* output);

  using prngenerator<TFloat>::_NUM_ENGINES;

  mersenne_twister* _prng_engines;
  uni_real_distribution* _prng_distributions;
};
}
#include "prngenerator_cpu_impl.hpp"
#endif /* _PRNGENERATOR_CPU_H_ */
