#ifndef LOCUSTA_PRNGENERATOR_H_
#define LOCUSTA_PRNGENERATOR_H_

#include <inttypes.h>

namespace locusta {

/**
 * @brief Abstract base class for pseudo-random number generators.
 *
 * This class defines the interface for pseudo-random number generators (PRNGs).
 * Concrete PRNG implementations, such as prngenerator_cpu and
 * prngenerator_cuda, derive from this abstract class.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct prngenerator
{
  /**
   * @brief Construct a new prngenerator object with one engine.
   */
  prngenerator()
    : _NUM_ENGINES(1)
  {
  }

  /**
   * @brief Construct a new prngenerator object.
   *
   * @param num_engines Number of PRNG engines to use.
   */
  prngenerator(uint32_t num_engines)
    : _NUM_ENGINES(num_engines)
  {
  }

  /**
   * @brief Destroy the prngenerator object.
   */
  virtual ~prngenerator() {}

  /**
   * @brief Initialize the state of the PRNG engines.
   *
   * @param seed Seed value for the PRNG engines.
   */
  virtual void _initialize_engines(uint64_t seed) = 0;

  /**
   * @brief Generate pseudo-random numbers.
   *
   * This method generates n pseudo-random numbers and stores them in the output
   * array.
   *
   * @param n Number of pseudo-random numbers to generate.
   * @param output Output array to store the generated pseudo-random numbers.
   */
  virtual void _generate(uint32_t n, TFloat* output) = 0;

  /**
   * @brief Number of PRNG engines.
   */
  const uint32_t _NUM_ENGINES;
};
}

#endif /* _PRNGENERATOR_H_ */
