#include "evaluator_cuda.hpp"

namespace locusta {
/**
 * @brief Construct a new evaluator_cuda object.
 *
 * @param eval_functor Evaluation functor.
 * @param f_negate Flag describing whether the evaluator will use the fitness
 * value or its negative.
 * @param bound_mapping_method Bound mapping method.
 * @param prn_numbers Number of pseudo random numbers needed per evaluation.
 */
template<typename TFloat>
evaluator_cuda<TFloat>::evaluator_cuda(
  EvaluationCudaFunctor<TFloat>* eval_functor,
  bool f_negate,
  BoundMapKind bound_mapping_method,
  uint32_t prn_numbers)
  : evaluator<TFloat>(eval_functor, f_negate, bound_mapping_method, prn_numbers)
{
  // Device Memory Allocation
  CudaSafeCall(
    cudaMalloc((void**)&_eval_prn_numbers, _eval_prn_size * sizeof(TFloat)));
}
/**
 * @brief Destroy the evaluator_cuda object.
 */
template<typename TFloat>
evaluator_cuda<TFloat>::~evaluator_cuda()
{
  CudaSafeCall(cudaFree(_eval_prn_numbers));
}
/**
 * @brief Evaluate the solver's population data set.
 *
 * This method evaluates the population data set by calling the evaluation
 * functor.
 *
 * @param solver Evolutionary solver.
 */
template<typename TFloat>
void
evaluator_cuda<TFloat>::evaluate(evolutionary_solver<TFloat>* solver)
{
  (*_evaluation_functor)(
    dynamic_cast<evolutionary_solver_cuda<TFloat>*>(solver));
}
/**
 * @brief Bound mapping implementation.
 *
 * This method implements the bound mapping for the CUDA architecture.
 *
 * @param bound_mapping_method Bound mapping method.
 * @param u Upper bound.
 * @param l Lower bound.
 * @param x Value to map.
 */
template<typename TFloat>
__device__ inline void
evaluator_cuda<TFloat>::bound_map(BoundMapKind bound_mapping_method,
                                  const TFloat& u,
                                  const TFloat& l,
                                  TFloat& x)
{
  float fx = static_cast<float>(x);
  float fu = static_cast<float>(u);
  float fl = static_cast<float>(l);
  if (fx == fx) {
    switch (bound_mapping_method) {
      case CropBounds: /// Out of bounds Crop
        x = fx > fu ? u : (fx < fl ? l : x);
        break;
      case MirrorBounds: /// Out of bounds Mirror
        x = fx > fu ? static_cast<TFloat>(2.0f * fu - fx) : (fx < fl ? static_cast<TFloat>(2.0f * fl - fx) : x);
        break;
      case IgnoreBounds: /// Out of bounds Error
        break;
    }
  } else {
    x = u;
  }
  return;
}

}
