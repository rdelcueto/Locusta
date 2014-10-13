#ifndef LOCUSTA_EVALUABLE_FUNCTION_H
#define LOCUSTA_EVALUABLE_FUNCTION_H

#include <limits>

namespace locusta {

    template<typename TFloat>
    struct evolutionary_solver;

    template<typename TFloat>
    struct EvaluationFunctor {
        virtual void operator()(evolutionary_solver<TFloat> * solver) {};
    };
}

#endif
