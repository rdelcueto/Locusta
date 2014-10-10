#ifndef LOCUSTA_PSO_STD_OPERATORS_H
#define LOCUSTA_PSO_STD_OPERATORS_H

#include "pso_operators.hpp"

namespace locusta {

    template<typename TFloat>
    struct CanonicalSpeedUpdate : UpdateSpeedFunctor<TFloat> {
        void operator()(pso_solver<TFloat> * solver)
            {
                std::cout << "UPDATE SPEED\n";
            }
    };

    template<typename TFloat>
    struct CanonicalPositionUpdate : UpdatePositionFunctor<TFloat> {
        void operator()(pso_solver<TFloat> * solver)
            {
                std::cout << "UPDATE POS\n";
            }
    };
}

#endif
