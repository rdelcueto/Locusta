#include "population_set.hpp"

namespace locusta {

/**
 * @brief Construct a new population_set object.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 */
template<typename TFloat>
population_set<TFloat>::population_set(const uint32_t ISLES,
                                       const uint32_t AGENTS,
                                       const uint32_t DIMENSIONS)
  : _ISLES(ISLES)
  , _AGENTS(AGENTS)
  , _DIMENSIONS(DIMENSIONS)
  , _GENES_PER_ISLE(_AGENTS * _DIMENSIONS)
  , _TOTAL_AGENTS(_AGENTS * _ISLES)
  , _TOTAL_GENES(_TOTAL_AGENTS * DIMENSIONS)
  , _f_initialized(0)
{
  // Host Memory allocation
  _data_array = new TFloat[_TOTAL_GENES];
  _transformed_data_array = new TFloat[_TOTAL_GENES];
  _fitness_array = new TFloat[_TOTAL_AGENTS];
}

/**
 * @brief Destroy the population_set object.
 */
template<typename TFloat>
population_set<TFloat>::~population_set()
{
  delete[] _fitness_array;
  delete[] _transformed_data_array;
  delete[] _data_array;
}

/**
 * @brief Swap the data sets.
 *
 * This method swaps the pointers between the current data set and the
 * transformed data set.
 */
template<typename TFloat>
void
population_set<TFloat>::swap_data_sets()
{
  TFloat* const swapped_ptr = _data_array;
  _data_array = _transformed_data_array;
  _transformed_data_array = swapped_ptr;
}
}
