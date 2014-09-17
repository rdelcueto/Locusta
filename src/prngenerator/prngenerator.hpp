#ifndef LOCUSTA_PRNGENERATOR_H_
#define LOCUSTA_PRNGENERATOR_H_

#include <inttypes.h>

namespace locusta {

  template<typename TFloat>
  class prngenerator {
  public:

    prngenerator(const uint32_t num_generators) : _NUM_GENERATORS(num_generators) {}

    virtual ~prngenerator() {}

    virtual void _initialize_engines(uint64_t seed) = 0;

    virtual void _generate(const uint32_t n, TFloat * output) = 0;

    virtual TFloat _generate() = 0;

  protected:

    const uint32_t _NUM_GENERATORS;

  };

}

#endif /* _PRNGENERATOR_H_ */
