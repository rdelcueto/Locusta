#ifndef LOCUSTA_EVALUATOR_H
#define LOCUSTA_EVALUATOR_H

#include <functional>

#include "../population/population_set.hpp"

namespace locusta {

    template<typename TFloat>
    class evaluator {
    public:

        enum bound_mapping_method_t
        {
            IGNORE_BOUNDS,
            CROP_BOUNDS,
            MIRROR_BOUNDS
        };

        evaluator(const bool f_negate,
                  const uint32_t bound_mapping_method,
                  const uint32_t num_eval_prnumbers)
            : _f_negate(f_negate),
              _bounding_mapping_method(bound_mapping_method),
              _num_eval_prnumbers(num_eval_prnumbers)
            {}

        virtual ~evaluator() {}

        virtual void evaluate(population_set<TFloat> * population,
                              const TFloat * UPPER_BOUNDS,
                              const TFloat * LOWER_BOUNDS) = 0;

        /// Flag describing whether the evaluator will the fitness value or its negative.
        const bool _f_negate;

        /// Specified the bounding map method.
        const uint32_t _bounding_mapping_method;

        /// Number of pseudo random numbers needed per evaluation
        const uint32_t _num_eval_prnumbers;

    };

} // namespace locusta

#endif
