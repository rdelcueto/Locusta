#ifndef _PRNGENERATOR_CUDA_H_
#define _PRNGENERATOR_CUDA_H_

#include "prngenerator.hpp"
#include <sys/time.h>

#include <curand.h>
#include <curand_kernel.h>

namespace locusta {

/**
 * @brief CUDA implementation of the pseudo-random number generator.
 *
 * This class implements the prngenerator interface for the CUDA architecture,
 * using the cuRAND library.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct prngenerator_cuda : prngenerator<TFloat>
{

  /**
   * @brief Construct a new prngenerator_cuda object with a single engine.
   */
  prngenerator_cuda();

  /**
   * @brief Construct a new prngenerator_cuda object.
   *
   * @param num_engines Number of PRNG engines to use.
   */
  prngenerator_cuda(uint32_t num_engines);

  /**
   * @brief Destroy the prngenerator_cuda object.
   */
  virtual ~prngenerator_cuda();

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

  /**
   * @brief Returns CUDA device engine states.
   *
   * @return curandState* Pointer to the array of CUDA device engine states.
   */
  virtual curandState* get_device_generator_states() const;

  using prngenerator<TFloat>::_NUM_ENGINES;

  /// CUDA bulk prng engine.
  curandGenerator_t _dev_bulk_prng_engine;

  /// CUDA prng kernel engines.
  curandState* _dev_prng_engines;
};

}

#include "prngenerator_cuda_impl.hpp"
#endif /* _PRNGENERATOR_CUDA_H_ */
