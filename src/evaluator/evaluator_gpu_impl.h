//TODO: REMOVE
#include <iostream>

#include "cuda_common/cuda_helpers.h"

namespace locusta {

    template<typename TFloat>
        evaluator_gpu<TFloat>::evaluator_gpu(const bool f_negate,
                                             const uint32_t bound_mapping_method,
                                             const uint32_t num_eval_prnumbers,
                                             gpu_evaluable_function fitness_function)
        : _fitness_function(fitness_function),
        evaluator<TFloat>(f_negate,
                          bound_mapping_method,
                          num_eval_prnumbers)
        {
        }

    template<typename TFloat>
        evaluator_gpu<TFloat>::~evaluator_gpu()
    {
    }

    template<typename TFloat>
        void evaluator_gpu<TFloat>::evaluate(population_set<TFloat> * population)
    {
        population_set_gpu<TFloat> * dev_population =
            static_cast<population_set_gpu<TFloat>*>(population);

        const TFloat * const UPPER_BOUNDS = dev_population->_get_dev_upper_bounds();
        const TFloat * const LOWER_BOUNDS = dev_population->_get_dev_lower_bounds();

        const TFloat * const data_array = dev_population->_get_dev_data_array();
        TFloat * const fitness_array = dev_population->_get_dev_fitness_array();

        _fitness_function(UPPER_BOUNDS,
                          LOWER_BOUNDS,
                          dev_population->_NUM_ISLES,
                          dev_population->_NUM_AGENTS,
                          dev_population->_NUM_DIMENSIONS,
                          _bounding_mapping_method,
                          _f_negate,
                          data_array,
                          fitness_array);
        return;
    }

} // namespace locusta
