#ifndef LOCUSTA_DE_OPERATORS_H
#define LOCUSTA_DE_OPERATORS_H

#include <inttypes.h>

namespace locusta {

  template<typename TFloat>
  struct de_solver_cpu;

  template<typename TFloat>
  struct DeBreedFunctor {
    virtual uint32_t required_prns(de_solver_cpu<TFloat> * solver) = 0;
    virtual void operator()(de_solver_cpu<TFloat> * solver) = 0;
  };

  template<typename TFloat>
  struct DeSelectionFunctor {
    virtual uint32_t required_prns(de_solver_cpu<TFloat> * solver) = 0;
    virtual void operator()(de_solver_cpu<TFloat> * solver) = 0;
  };
}

#endif
