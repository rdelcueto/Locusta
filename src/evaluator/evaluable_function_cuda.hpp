#ifndef LOCUSTA_EVALUABLE_FUNCTION_CUDA_H
#define LOCUSTA_EVALUABLE_FUNCTION_CUDA_H

#include "evaluable_function.hpp"
#include "math_constants.h"

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
 * @brief Abstract base class for CUDA evaluation functors.
 *
 * This class defines the interface for CUDA evaluation functors, which are used
 * to evaluate the fitness of a population of candidate solutions on the GPU.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct EvaluationCudaFunctor : EvaluationFunctor<TFloat>
{
  /**
   * @brief Function call operator.
   *
   * This operator is used to evaluate the fitness of a population of candidate
   * solutions on the GPU.
   *
   * @param solver CUDA evolutionary solver.
   */
  virtual void operator()(evolutionary_solver_cuda<TFloat>* solver) {};
};
}

#endif
