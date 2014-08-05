#ifndef LOCUSTA_EVALUATOR_CPU_H
#define LOCUSTA_EVALUATOR_CPU_H

#include "evaluator.h"

namespace locusta {

    template<typename TFloat>
        class evaluator_cpu : public evaluator<TFloat> {
    public:

        typedef void (*cpu_evaluable_function)(const TFloat * const UPPER_BOUNDS,
                                               const TFloat * const LOWER_BOUNDS,
                                               const uint32_t NUM_ISLES,
                                               const uint32_t NUM_AGENTS,
                                               const uint32_t NUM_DIMENSIONS,
                                               const uint32_t bound_mapping_method,
                                               const bool f_negate,
                                               const TFloat * const agents_data,
                                               TFloat * const agents_fitness);

        evaluator_cpu(const bool f_negate,
                      const uint32_t bound_mapping_method,
                      const uint32_t num_eval_prnumbers,
                      cpu_evaluable_function fitness_function);

        virtual ~evaluator_cpu();

        virtual void evaluate(population_set<TFloat> * population);

        using evaluator<TFloat>::_f_negate;
        using evaluator<TFloat>::_bounding_mapping_method;
        using evaluator<TFloat>::_num_eval_prnumbers;

        // Pointer to fitness function
        cpu_evaluable_function _fitness_function;
    };

} // namespace locusta
#include "evaluator_cpu.cpp"
#endif
