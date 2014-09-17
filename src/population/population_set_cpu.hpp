#ifndef LOCUSTA_POPULATION_SET_CPU_H_
#define LOCUSTA_POPULATION_SET_CPU_H_

#include <iostream>
#include <limits>

#include <omp.h>

#include "population_set.hpp"

namespace locusta {

  template <typename TFloat>
  class population_set_cpu : public population_set<TFloat> {
  public:
    population_set_cpu(const uint32_t NUM_ISLES,
                       const uint32_t NUM_AGENTS,
                       const uint32_t NUM_DIMENSIONS,
                       TFloat * upper_bound,
                       TFloat * lower_bound);

    virtual ~population_set_cpu();

    virtual void _initialize();
    virtual void _print_data();
    virtual void _swap_data_sets();
    virtual void _update_records();

    using population_set<TFloat>::_get_data_array;
    using population_set<TFloat>::_get_transformed_data_array;

    using population_set<TFloat>::_get_fitness_array;

    using population_set<TFloat>::_get_upper_bounds;
    using population_set<TFloat>::_get_lower_bounds;

    using population_set<TFloat>::_get_global_highest_isle_idx;
    using population_set<TFloat>::_get_global_highest_fitness;
    using population_set<TFloat>::_get_global_lowest_isle_idx;
    using population_set<TFloat>::_get_global_lowest_fitness;

    using population_set<TFloat>::_get_highest_idx_array;
    using population_set<TFloat>::_get_highest_fitness_array;

    using population_set<TFloat>::_get_lowest_idx_array;
    using population_set<TFloat>::_get_lowest_fitness_array;


    using population_set<TFloat>::_NUM_ISLES;
    using population_set<TFloat>::_NUM_AGENTS;
    using population_set<TFloat>::_NUM_DIMENSIONS;

    using population_set<TFloat>::_UPPER_BOUNDS;
    using population_set<TFloat>::_LOWER_BOUNDS;

    using population_set<TFloat>::_var_ranges;

  protected:

    using population_set<TFloat>::_data_array;
    using population_set<TFloat>::_transformed_data_array;
    using population_set<TFloat>::_fitness_array;

    using population_set<TFloat>::_highest_idx;
    using population_set<TFloat>::_highest_fitness;

    using population_set<TFloat>::_lowest_idx;
    using population_set<TFloat>::_lowest_fitness;

    using population_set<TFloat>::_global_highest_idx;
    using population_set<TFloat>::_global_highest_fitness;

    using population_set<TFloat>::_global_lowest_idx;
    using population_set<TFloat>::_global_lowest_fitness;

  };

}
#include "population_set_cpu.cpp"
#endif /* LOCUSTA_POPULATION_SET_CPU_H_ */
