#ifndef LOCUSTA_EVALUATOR_CPU_H
#define LOCUSTA_EVALUATOR_CPU_H

#include <functional>

#include "evaluator.hpp"

namespace locusta {

  template<typename TFloat>
  struct evolutionary_solver_cpu;

  template<typename TFloat>
  struct evaluator_cpu : evaluator<TFloat> {
    evaluator_cpu(EvaluationFunctor<TFloat> * eval_functor,
                  bool f_negate,
                  BoundMapKind bound_mapping,
                  uint32_t prn_numbers);

    virtual ~evaluator_cpu();

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
#include "evaluator_cpu_impl.hpp"
#endif
