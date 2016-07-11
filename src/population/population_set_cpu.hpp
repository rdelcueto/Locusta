#ifndef LOCUSTA_POPULATION_SET_CPU_H_
#define LOCUSTA_POPULATION_SET_CPU_H_

#include <iostream>
#include <limits>

#include "population_set.hpp"

namespace locusta {

  template <typename TFloat>
  struct population_set_cpu : population_set<TFloat> {

    population_set_cpu(const uint32_t ISLES,
                       const uint32_t AGENTS,
                       const uint32_t DIMENSIONS)
      : population_set<TFloat>(ISLES,
                               AGENTS,
                               DIMENSIONS) {}

    virtual ~population_set_cpu() {};

  };
}

#endif /* LOCUSTA_POPULATION_SET_CPU_H_ */
