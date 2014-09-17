#ifndef LOCUSTA_EVALUATOR_GPU_H
#define LOCUSTA_EVALUATOR_GPU_H

#include "evaluator.hpp"
#include "../population/population_set_gpu.hpp"

namespace locusta {

  template<typename TFloat>
  class evaluator_gpu : public evaluator<TFloat> {
  public:

    typedef void (*gpu_evaluable_function)(const TFloat * const UPPER_BOUNDS,
                                           const TFloat * const LOWER_BOUNDS,
                                           const uint32_t NUM_ISLES,
                                           const uint32_t NUM_AGENTS,
                                           const uint32_t NUM_DIMENSIONS,
                                           const uint32_t bound_mapping_method,
                                           const bool f_negate,
                                           const TFloat * const agents_data,
                                           TFloat * const agents_fitness);

    evaluator_gpu(const bool f_negate,
                  const uint32_t bound_mapping_method,
                  const uint32_t num_eval_prnumbers,
                  gpu_evaluable_function fitness_function);

    virtual ~evaluator_gpu();

    virtual void evaluate(population_set<TFloat> * population);

    using evaluator<TFloat>::_f_negate;
    using evaluator<TFloat>::_bounding_mapping_method;
    using evaluator<TFloat>::_num_eval_prnumbers;

    /// Pointer to fitness function
    gpu_evaluable_function _fitness_function;
  };

} /// namespace locusta
#include "evaluator_gpu_impl.hpp"
#endif
