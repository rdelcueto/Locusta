#ifndef LOCUSTA_EVALUATOR_CUDA_H
#define LOCUSTA_EVALUATOR_CUDA_H

#include <functional>

#include "cuda_common/cuda_helpers.h"

#include "evaluable_function_cuda.hpp"
#include "evaluator.hpp"

namespace locusta {
/**
 * @brief Forward declaration of the evolutionary_solver_cuda type.
 *
 * The evolutionary_solver_cuda class is a specific implementation of the
 * evolutionary_solver for the CUDA architecture.
 */
template<typename TFloat>
struct evolutionary_solver_cuda;
/**
 * @brief CUDA evaluator type.
 *
 * This class implements the evaluator interface for the CUDA architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct evaluator_cuda : evaluator<TFloat>
{

  /**
   * @brief Construct a new evaluator_cuda object.
   *
   * @param eval_functor Evaluation functor.
   * @param f_negate Flag describing whether the evaluator will use the fitness
   * value or its negative.
   * @param bound_mapping Bound mapping method.
   * @param prn_numbers Number of pseudo random numbers needed per evaluation.
   */
  evaluator_cuda(EvaluationCudaFunctor<TFloat>* eval_functor,
                 bool f_negate,
                 BoundMapKind bound_mapping,
                 uint32_t prn_numbers);
  /**
   * @brief Destroy the evaluator_cuda object.
   */
  virtual ~evaluator_cuda();

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
#include "evaluator_cuda_impl.hpp"
#endif
