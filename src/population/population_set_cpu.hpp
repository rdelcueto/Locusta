#ifndef LOCUSTA_POPULATION_SET_CPU_H_
#define LOCUSTA_POPULATION_SET_CPU_H_

#include <iostream>
#include <limits>

#include "population_set.hpp"

namespace locusta {

/**
 * @brief CPU implementation of the population_set class.
 *
 * This class extends the population_set class with CPU-specific functionality.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct population_set_cpu : population_set<TFloat>
{

  /**
   * @brief Construct a new population_set_cpu object.
   *
   * @param ISLES Number of isles in the population.
   * @param AGENTS Number of agents per isle.
   * @param DIMENSIONS Number of dimensions per agent.
   */
  population_set_cpu(const uint32_t ISLES,
                     const uint32_t AGENTS,
                     const uint32_t DIMENSIONS)
    : population_set<TFloat>(ISLES, AGENTS, DIMENSIONS)
  {
  }

  /**
   * @brief Destroy the population_set_cpu object.
   */
  virtual ~population_set_cpu(){};
};
}

#endif /* LOCUSTA_POPULATION_SET_CPU_H_ */
