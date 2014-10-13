#ifndef LOCUSTA_EVALUATOR_H
#define LOCUSTA_EVALUATOR_H

#include <functional>

#include "evaluable_function.hpp"

namespace locusta {

    template<typename TFloat>
    struct evolutionary_solver;

    enum class BoundMapKind : uint8_t { IgnoreBounds,
            CropBounds, MirrorBounds };

    template<typename TFloat>
    struct evaluator {
        evaluator(EvaluationFunctor<TFloat> * eval_functor,
                  bool f_negate,
                  BoundMapKind bound_mapping_method,
                  uint32_t prn_numbers);

        virtual ~evaluator();

        /// Evaluate the solver's population data set.
        virtual void evaluate(evolutionary_solver<TFloat> * solver);

        /// Bound mapping implementation
        static void bound_map(BoundMapKind bound_mapping_method,
                              const TFloat &u,
                              const TFloat &l,
                              TFloat &x);

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

#include "evaluator.cpp"

#endif
