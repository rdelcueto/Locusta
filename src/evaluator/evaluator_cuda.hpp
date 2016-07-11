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

    virtual void evaluate(evolutionary_solver<TFloat> * solver);

    virtual void bound_map(BoundMapKind bound_mapping_method,
                           const TFloat &u,
                           const TFloat &l,
                           TFloat &x);

    using evaluator<TFloat>::_f_negate;
    using evaluator<TFloat>::_bound_mapping_method;
    using evaluator<TFloat>::_eval_prn_size;
    using evaluator<TFloat>::_eval_prn_numbers;
    using evaluator<TFloat>::_evaluation_functor;
  };

} // namespace locusta
#include "evaluator_cuda_impl.hpp"
#endif
