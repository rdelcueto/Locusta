#ifndef LOCUSTA_POPULATION_SET_H_
#define LOCUSTA_POPULATION_SET_H_

#include <inttypes.h>

namespace locusta {

  template <typename TFloat>
  class population_set {
  public:

    population_set(const uint32_t NUM_ISLES,
                   const uint32_t NUM_AGENTS,
                   const uint32_t NUM_DIMENSIONS,
                   TFloat * upper_bounds,
                   TFloat * lower_bounds)
      : _NUM_ISLES(NUM_ISLES),
        _NUM_AGENTS(NUM_AGENTS),
        _NUM_DIMENSIONS(NUM_DIMENSIONS),
        _UPPER_BOUNDS(upper_bounds),
        _LOWER_BOUNDS(lower_bounds)
    {}

    virtual ~population_set() {}

    /// Initializes values. Transformed data array must have Uniform pseudo random real numbers [0,1).
    virtual void _initialize() = 0;
    /// Print Raw Data
    virtual void _print_data() = 0;
    /// Swaps pointers of current data set with temporal data set.
    virtual void _swap_data_sets() = 0;
    /// Updates records using latest evaluation values.
    virtual void _update_records() = 0;

    /// Returns the isle's index with the highest fitness agent.
    virtual uint32_t _get_global_highest_isle_idx()
    {
      return _global_highest_idx;
    }

    /// Returns the highest fitness value.
    virtual TFloat _get_global_highest_fitness()
    {
      return _global_highest_fitness;
    }

    /// Returns the isle's index with the lowest fitness agent.
    virtual uint32_t _get_global_lowest_isle_idx()
    {
      return _global_lowest_idx;
    }

    /// Returns the lowest fitness value.
    virtual TFloat _get_global_lowest_fitness()
    {
      return _global_lowest_fitness;
    }

    /// Returns a const pointer to the population's _highest_idx array.
    virtual uint32_t * _get_highest_idx_array()
    {
      return _highest_idx;
    }

    /// Returns a const pointer to the population's _highest_fitness_array.
    virtual TFloat * _get_highest_fitness_array()
    {
      return _highest_fitness;
    }

    /// Returns a const pointer to the population's _lowest_idx array.
    virtual uint32_t * _get_lowest_idx_array()
    {
      return _lowest_idx;
    }

    /// Returns a const pointer to the population's _highest_fitness_array.
    virtual TFloat * _get_lowest_fitness_array()
    {
      return _lowest_fitness;
    }

    /// Returns a pointer to the raw data array.
    virtual TFloat * _get_data_array()
    {
      return _data_array;
    }

    /// Returns a pointer to the raw transformed data array.
    virtual TFloat * _get_transformed_data_array()
    {
      return _transformed_data_array;
    }

    /// Returns a pointer to the raw fitness array.
    virtual TFloat * _get_fitness_array()
    {
      return _fitness_array;
    }

    /// Returns a pointer to the upper bounds array.
    virtual const TFloat * _get_upper_bounds()
    {
      return _UPPER_BOUNDS;
    }

    /// Returns a pointer to the lower bounds array.
    virtual const TFloat * _get_lower_bounds()
    {
      return _LOWER_BOUNDS;
    }

    /// Describes the number of isles
    const uint32_t _NUM_ISLES;
    /// Describes the number of agents in every isle.
    const uint32_t _NUM_AGENTS;
    /// Describes the number of encoded variables per agent.
    const uint32_t _NUM_DIMENSIONS;

    /// Pointer to array describing the upper bound of each variable.
    const TFloat * const _UPPER_BOUNDS;
    /// Pointer to array describing the lower bound of each variable.
    const TFloat * const _LOWER_BOUNDS;
    /// Pointer to array describing the range of each variable.
    TFloat * _var_ranges;

  protected:

    /// Pointer to the agent's encoded data variables.
    TFloat * _data_array;
    /// Pointer to the agent's encoded data variables.
    TFloat * _transformed_data_array;
    /// Pointer to the agent's fitness value.
    TFloat * _fitness_array;

    /// Describes the index of the isle with the highest fitness among the whole population.
    uint32_t _global_highest_idx;
    /// Describes the index of the isle with the lowest fitness among the whole population.
    uint32_t _global_lowest_idx;

    /// Describes the highest fitness among the whole population.
    TFloat _global_highest_fitness;
    /// Describes the lowest fitness among the whole population.
    TFloat _global_lowest_fitness;

    /// Describes the index of the isle's highest fitness agents.
    uint32_t * _highest_idx;
    /// Describes the isle's highest fitness values.
    TFloat * _highest_fitness;

    /// Describes the index of the isle's lowest fitness agents.
    uint32_t * _lowest_idx;
    /// Describes the isle's lowest fitness values.
    TFloat * _lowest_fitness;
  };

}

#endif /* LOCUSTA_POPULATION_SET_H_ */
