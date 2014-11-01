#ifndef LOCUSTA_EVALUATOR_H
#define LOCUSTA_EVALUATOR_H

#include <functional>
#include <stdint.h>

#include "evaluable_function.hpp"

namespace locusta {

    template<typename TFloat>
    struct evolutionary_solver;

    enum BoundMapKind { IgnoreBounds,
                        CropBounds, MirrorBounds };

    template<typename TFloat>
    struct evaluator {
        evaluator(EvaluationFunctor<TFloat> * eval_functor,
                  bool f_negate,
                  BoundMapKind bound_mapping_method,
                  uint32_t prn_numbers)
            : _evaluation_functor(eval_functor),
              _f_negate(f_negate),
              _bound_mapping_method(bound_mapping_method),
              _eval_prn_size(prn_numbers) {}

        virtual ~evaluator() {}

        /// Evaluate the solver's population data set.
        virtual void evaluate(evolutionary_solver<TFloat> * solver) = 0;

        /// Bound mapping implementation
        virtual void bound_map(BoundMapKind bound_mapping_method,
                              const TFloat &u,
                              const TFloat &l,
                              TFloat &x) = 0;

        /// Flag describing whether the evaluator will the fitness value or its negative.
        const bool _f_negate;

        /// Specified the bounding map method.
        const BoundMapKind _bound_mapping_method;

        /// Number of pseudo random numbers needed per evaluation.
        const size_t _eval_prn_size;

        /// Evaluation random numbers array.
        TFloat * _eval_prn_numbers;

        /// Evaluation dispatch functor
        EvaluationFunctor<TFloat> * _evaluation_functor;

    };

} // namespace locusta

#endif
