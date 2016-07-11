#ifndef LOCUSTA_EVALUABLE_FUNCTION_CUDA_H
#define LOCUSTA_EVALUABLE_FUNCTION_CUDA_H

#include "evaluable_function.hpp"
#include "math_constants.h"

namespace locusta {

  template<typename TFloat>
  struct evolutionary_solver_cuda;

  template<typename TFloat>
  struct EvaluationCudaFunctor : EvaluationFunctor<TFloat>{
    virtual void operator()(evolutionary_solver_cuda<TFloat> * solver) {};
  };
}

#endif
