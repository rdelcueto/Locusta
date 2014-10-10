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
    void evaluator_gpu<TFloat>::evaluate(population_set<TFloat> * population,
                                         const TFloat * UPPER_BOUNDS,
                                         const TFloat * LOWER_BOUNDS)
    {
        population_set_gpu<TFloat> * dev_population =
            static_cast<population_set_gpu<TFloat>*>(population);

        _fitness_function(UPPER_BOUNDS,
                          LOWER_BOUNDS,
                          dev_population->_ISLES,
                          dev_population->_AGENTS,
                          dev_population->_DIMENSIONS,
                          _bounding_mapping_method,
                          _f_negate,
                          dev_population->_dev_data_array,
                          dev_population->_dev_fitness_array);
        return;
    }

} // namespace locusta
