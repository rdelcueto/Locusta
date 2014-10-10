#ifndef LOCUSTA_PSO_OPERATORS_H
#define LOCUSTA_PSO_OPERATORS_H

namespace locusta {

    template<typename TFloat>
    struct pso_solver;

    template<typename TFloat>
    struct UpdateSpeedFunctor {
        virtual void operator()(pso_solver<TFloat> * solver) {};
    };

    template<typename TFloat>
    struct UpdatePositionFunctor {
        virtual void operator()(pso_solver<TFloat> * solver) {};
    };
}

#endif
