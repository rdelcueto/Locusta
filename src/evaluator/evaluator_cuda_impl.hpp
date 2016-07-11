#include "evaluator_cuda.hpp"

namespace locusta {

  template<typename TFloat>
  evaluator_cuda<TFloat>::evaluator_cuda(EvaluationCudaFunctor<TFloat> * eval_functor,
                                         bool f_negate,
                                         BoundMapKind bound_mapping_method,
                                         uint32_t prn_numbers) :
    evaluator<TFloat>(eval_functor,
                      f_negate,
                      bound_mapping_method,
                      prn_numbers)
  {
    // Device Memory Allocation
    CudaSafeCall(cudaMalloc((void **) &_eval_prn_numbers, _eval_prn_size * sizeof(TFloat)));
  }

  template<typename TFloat>
  evaluator_cuda<TFloat>::~evaluator_cuda()
  {
    CudaSafeCall(cudaFree(_eval_prn_numbers));
  }

  template<typename TFloat>
  void evaluator_cuda<TFloat>::evaluate(evolutionary_solver<TFloat> * solver)
  {
    (*_evaluation_functor)(dynamic_cast<evolutionary_solver_cuda<TFloat> *>(solver));
  }

  template<typename TFloat>
  __device__
  inline void evaluator_cuda<TFloat>::bound_map(BoundMapKind bound_mapping_method,
                                                const TFloat &u,
                                                const TFloat &l,
                                                TFloat &x)
  {
    if (x == x)
      {
        switch (bound_mapping_method)
          {
          case CropBounds: /// Out of bounds Crop
            x = x > u ? u : x < l ? l : x;
            break;
          case MirrorBounds: /// Out of bounds Mirror
            x = x > u ? (2*u - x) : x < l ? (2*l - x) : x;
            break;
          case IgnoreBounds: /// Out of bounds Error
            break;
          }
      }
    else
      {
        x = u;
      }
    return;
  }

}
