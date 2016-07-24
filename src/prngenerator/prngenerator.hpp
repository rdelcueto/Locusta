#ifndef LOCUSTA_PRNGENERATOR_H_
#define LOCUSTA_PRNGENERATOR_H_

#include <inttypes.h>

namespace locusta {

  template<typename TFloat>
  struct prngenerator {

    prngenerator() : _NUM_ENGINES(1) {}
    prngenerator(uint32_t num_engines) : _NUM_ENGINES(num_engines) {}
    virtual ~prngenerator() {}

    /// Initialize engines state
    virtual void _initialize_engines(uint64_t seed) = 0;

    /// Generate n pseudo random numbers into output array.
    virtual void _generate(uint32_t n, TFloat * output) = 0;

    /// Number of prng engines.
    const uint32_t _NUM_ENGINES;
  };
}

#endif /* _PRNGENERATOR_H_ */
