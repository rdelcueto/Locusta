#include <iostream> // TODO: REMOVE

namespace locusta {


    template<typename TFloat>
    evaluator_cpu<TFloat>::evaluator_cpu(const bool f_negate,
                                         const uint32_t bound_mapping_method,
                                         const uint32_t num_eval_prnumbers,
                                         cpu_evaluable_function fitness_function)
        : _fitness_function(fitness_function),
          evaluator<TFloat>(f_negate,
                            bound_mapping_method,
                            num_eval_prnumbers)
    {
    }

    template<typename TFloat>
    evaluator_cpu<TFloat>::~evaluator_cpu()
    {
    }

    template<typename TFloat>
    void evaluator_cpu<TFloat>::evaluate(population_set<TFloat> * population)
    {
        const TFloat * const data_array = population->_get_data_array();
        TFloat * const fitness_array = population->_get_fitness_array();

        _fitness_function(population->_UPPER_BOUNDS,
                          population->_LOWER_BOUNDS,
                          population->_NUM_ISLES,
                          population->_NUM_AGENTS,
                          population->_NUM_DIMENSIONS,
                          _bounding_mapping_method,
                          _f_negate,
                          data_array,
                          fitness_array);
        return;
    }

} // namespace locusta
