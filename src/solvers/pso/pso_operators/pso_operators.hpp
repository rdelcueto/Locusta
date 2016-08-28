#ifndef LOCUSTA_PSO_OPERATORS_H
#define LOCUSTA_PSO_OPERATORS_H

#include <inttypes.h>

namespace locusta {

template <typename TFloat>
struct pso_solver_cpu;

template <typename TFloat>
struct UpdateSpeedFunctor
{
  virtual uint32_t required_prns(pso_solver_cpu<TFloat>* solver) = 0;
  virtual void operator()(pso_solver_cpu<TFloat>* solver) = 0;
};

template <typename TFloat>
struct UpdateParticleRecordFunctor
{
  virtual uint32_t required_prns(pso_solver_cpu<TFloat>* solver) = 0;
  virtual void operator()(pso_solver_cpu<TFloat>* solver) = 0;
};

template <typename TFloat>
struct UpdatePositionFunctor
{
  virtual uint32_t required_prns(pso_solver_cpu<TFloat>* solver) = 0;
  virtual void operator()(pso_solver_cpu<TFloat>* solver) = 0;
};
}

#endif
