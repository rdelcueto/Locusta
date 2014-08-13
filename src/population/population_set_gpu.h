#ifndef LOCUSTA_POPULATION_SET_GPU_H_
#define LOCUSTA_POPULATION_SET_GPU_H_

#include <iostream>
#include <limits>
#include "math_constants.h"
#include "population_set.h"

namespace locusta {

    template <typename TFloat>
        class population_set_gpu :
        public population_set<TFloat> {

    public:
        population_set_gpu(const uint32_t NUM_ISLES,
                           const uint32_t NUM_AGENTS,
                           const uint32_t NUM_DIMENSIONS,
                           TFloat * upper_bounds,
                           TFloat * lower_bounds);

        virtual ~population_set_gpu();

        virtual void _initialize();
        virtual void _print_data();
        virtual void _swap_data_sets();
        virtual void _update_records();

        using population_set<TFloat>::_get_data_array;
        using population_set<TFloat>::_get_transformed_data_array;

        using population_set<TFloat>::_get_fitness_array;

        using population_set<TFloat>::_get_upper_bounds;
        using population_set<TFloat>::_get_lower_bounds;

        virtual TFloat * _get_dev_data_array();
        virtual TFloat * _get_dev_transformed_data_array();

        virtual TFloat * _get_dev_fitness_array();

        virtual const TFloat * _get_dev_upper_bounds();
        virtual const TFloat * _get_dev_lower_bounds();

        virtual void _copy_dev_data_into_host(TFloat * const output_data_array);
        virtual void _copy_host_data_into_dev(const TFloat * const input_data_array);

        virtual void _copy_dev_fitness_into_host(TFloat * const output_fitness_array);
        virtual void _copy_host_fitness_into_dev(const TFloat * const input_fitness_array);

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

        /// Device pointer to array describing the upper bound of each variable.
        TFloat * _DEV_UPPER_BOUNDS;
        /// Device pointer to array describing the lower bound of each variable.
        TFloat * _DEV_LOWER_BOUNDS;
        /// Device pointer to array describing the range of each variable.
        TFloat * _dev_var_ranges;

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

        /// Device pointer to the agent's encoded data variables.
        TFloat * _dev_data_array;
        /// Device pointer to the agent's encoded data variables.
        TFloat * _dev_transformed_data_array;
        /// Device pointer to the agent's fitness value.
        TFloat * _dev_fitness_array;

        /// Device pointer to index of the isle's highest fitness agents.
        uint32_t * _dev_highest_idx;
        /// Device pointer to the isle's highest fitness values.
        TFloat * _dev_highest_fitness;

        /// Device pointer to the index of the isle's lowest fitness agents.
        uint32_t * _dev_lowest_idx;
        /// Device pointer to the isle's lowest fitness values.
        TFloat * _dev_lowest_fitness;

    };
}
#include "population_set_gpu_impl.h"

#endif /* LOCUSTA_POPULATION_SET_GPU_H_ */
