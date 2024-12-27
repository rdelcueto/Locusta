#ifndef LOCUSTA_POPULATION_SET_H_
#define LOCUSTA_POPULATION_SET_H_

#include <inttypes.h>

namespace locusta {

/**
 * @brief Class for managing a population of candidate solutions.
 *
 * This class represents a population of candidate solutions, organized into
 * isles, agents, and dimensions. It provides methods for managing the
 * population data and performing operations on it.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct population_set
{

  /**
   * @brief Construct a new population_set object.
   *
   * @param ISLES Number of isles in the population.
   * @param AGENTS Number of agents per isle.
   * @param DIMENSIONS Number of dimensions per agent.
   */
  population_set(const uint32_t ISLES,
                 const uint32_t AGENTS,
                 const uint32_t DIMENSIONS);

  /**
   * @brief Destroy the population_set object.
   */
  virtual ~population_set();

  /**
   * @brief Swap the data sets.
   *
   * This method swaps the pointers between the current data set and the
   * transformed data set.
   */
  virtual void swap_data_sets();

  /// Number of Isles
  const uint32_t _ISLES;
  /// Number of Agents per isle
  const uint32_t _AGENTS;
  /// Number of dimensions per agent
  const uint32_t _DIMENSIONS;
  /// Number of genes per isle. Agents * Dimensions.
  const uint32_t _GENES_PER_ISLE;
  /// Number of total genes. Isles * Agents.
  const uint32_t _TOTAL_AGENTS;
  /// Number of total genes. Isles * Agents * Dimensions.
  const uint32_t _TOTAL_GENES;

  /// Genomes array.
  TFloat* _data_array;
  /// Transformed Genomes array.
  TFloat* _transformed_data_array;
  /// Genome Fitness array.
  TFloat* _fitness_array;

  /// Initialize flag
  uint8_t _f_initialized;
};
}
#include "population_set_impl.hpp"
#endif /* LOCUSTA_POPULATION_SET_H_ */
