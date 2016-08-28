#ifndef LOCUSTA_DE_OPERATORS_CUDA_H
#define LOCUSTA_DE_OPERATORS_CUDA_H

#include <inttypes.h>

namespace locusta {

template <typename TFloat>
struct de_solver_cuda;

template <typename TFloat>
struct DeBreedCudaFunctor
{
  virtual uint32_t required_prns(de_solver_cuda<TFloat>* solver) = 0;
  virtual void operator()(de_solver_cuda<TFloat>* solver) = 0;
};

template <typename TFloat>
struct DeSelectionCudaFunctor
{
  virtual uint32_t required_prns(de_solver_cuda<TFloat>* solver) = 0;
  virtual void operator()(de_solver_cuda<TFloat>* solver) = 0;
};
}

#endif
