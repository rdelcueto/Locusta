#ifndef LOCUSTA_EVALUATOR_CUDA_H
#define LOCUSTA_EVALUATOR_CUDA_H

#include <functional>

#include "cuda_common/cuda_helpers.h"

#include "evaluator.hpp"
#include "evaluable_function_cuda.hpp"

namespace locusta {

    template<typename TFloat>
    struct evolutionary_solver_cuda;

    template<typename TFloat>
    struct evaluator_cuda : evaluator<TFloat> {
        evaluator_cuda(EvaluationCudaFunctor<TFloat> * eval_functor,
                       bool f_negate,
                       BoundMapKind bound_mapping,
                       uint32_t prn_numbers);

        virtual ~evaluator_cuda();

        /// Evaluate the solver's population data set.
        virtual void evaluate(evolutionary_solver_cuda<TFloat> * solver);

        /// Bound mapping implementation
        static void bound_map(BoundMapKind bound_mapping_method,
                              const TFloat &u,
                              const TFloat &l,
                              TFloat &x);

       /// Flag describing whether the evaluator will the fitness value or its negative.
        const bool _f_negate;

        /// Specified the bounding map method.
        const BoundMapKind _bound_mapping;

        /// Number of pseudo random numbers needed per evaluation.
        const size_t _eval_prn_size;

        /// Evaluation random numbers array.
        TFloat * _eval_prn_numbers;

        /// Evaluation dispatch functor
        EvaluationCudaFunctor<TFloat> * _evaluation_functor;



    };

} // namespace locusta

#endif
