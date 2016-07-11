#ifndef LOCUSTA_GA_OPERATORS_CUDA_H
#define LOCUSTA_GA_OPERATORS_CUDA_H

#include <inttypes.h>

namespace locusta {

  template<typename TFloat>
  struct ga_solver_cuda;

  template<typename TFloat>
  struct BreedCudaFunctor {
    virtual uint32_t required_prns(ga_solver_cuda<TFloat> * solver) = 0;
    virtual void operator()(ga_solver_cuda<TFloat> * solver) = 0;
  };

  template<typename TFloat>
  struct SelectionCudaFunctor {
    virtual uint32_t required_prns(ga_solver_cuda<TFloat> * solver) = 0;
    virtual void operator()(ga_solver_cuda<TFloat> * solver) = 0;
  };
}

#endif
