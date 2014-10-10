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
  void evaluator_cpu<TFloat>::evaluate(population_set<TFloat> * population,
                                       const TFloat * UPPER_BOUNDS,
                                       const TFloat * LOWER_BOUNDS)
  {

    _fitness_function(UPPER_BOUNDS,
                      LOWER_BOUNDS,
                      population->_ISLES,
                      population->_AGENTS,
                      population->_DIMENSIONS,
                      _bounding_mapping_method,
                      _f_negate,
                      population->_data_array,
                      population->_fitness_array);
    return;
  }

} // namespace locusta
