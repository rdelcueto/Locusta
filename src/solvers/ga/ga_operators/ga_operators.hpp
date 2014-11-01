#ifndef LOCUSTA_GA_OPERATORS_H
#define LOCUSTA_GA_OPERATORS_H

#include <inttypes.h>

namespace locusta {

    template<typename TFloat>
    struct ga_solver_cpu;

    template<typename TFloat>
    struct BreedFunctor {
        virtual uint32_t required_prns(ga_solver_cpu<TFloat> * solver) = 0;
        virtual void operator()(ga_solver_cpu<TFloat> * solver) = 0;
    };

    template<typename TFloat>
    struct SelectionFunctor {
        virtual uint32_t required_prns(ga_solver_cpu<TFloat> * solver) = 0;
        virtual void operator()(ga_solver_cpu<TFloat> * solver) = 0;
    };
}

#endif
