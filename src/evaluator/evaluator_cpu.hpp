#ifndef LOCUSTA_EVALUATOR_CPU_H
#define LOCUSTA_EVALUATOR_CPU_H

#include <functional>

#include "evaluator.hpp"

namespace locusta {

/**
 * @brief Forward declaration of the evolutionary_solver_cpu type.
 *
 * The evolutionary_solver_cpu class is a specific implementation of the
 * evolutionary_solver for the CPU architecture.
 */
template<typename TFloat>
struct evolutionary_solver_cpu;

/**
 * @brief CPU evaluator type.
 *
 * This class implements the evaluator interface for the CPU architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct evaluator_cpu : evaluator<TFloat>
{
  /**
   * @brief Construct a new evaluator_cpu object.
   *
   * @param eval_functor Evaluation functor.
   * @param f_negate Flag describing whether the evaluator will use the
   * fitness value or its negative.
   * @param bound_mapping Bound mapping method.
   * @param prn_numbers Number of pseudo random numbers needed per
   * evaluation.
   */
  evaluator_cpu(EvaluationFunctor<TFloat>* eval_functor,
                bool f_negate,
                BoundMapKind bound_mapping,
                uint32_t prn_numbers);

  /**
   * @brief Destroy the evaluator_cpu object.
   */
  virtual ~evaluator_cpu();

  /**
   * @brief Evaluate the solver's population data set.
   *
   * @param solver Evolutionary solver.
   */
  virtual void evaluate(evolutionary_solver<TFloat>* solver);

  /**
   * @brief Bound mapping implementation.
   *
   * @param bound_mapping_method Bound mapping method.
   * @param u Upper bound.
   * @param l Lower bound.
   * @param x Value to map.
   */
  virtual void bound_map(BoundMapKind bound_mapping_method,
                         const TFloat& u,
                         const TFloat& l,
                         TFloat& x);

  using evaluator<TFloat>::_f_negate;
  using evaluator<TFloat>::_bound_mapping_method;
  using evaluator<TFloat>::_eval_prn_size;
  using evaluator<TFloat>::_eval_prn_numbers;
  using evaluator<TFloat>::_evaluation_functor;
};

} // namespace locusta
#include "evaluator_cpu_impl.hpp"
#endif
