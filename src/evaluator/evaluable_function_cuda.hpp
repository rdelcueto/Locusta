#ifndef LOCUSTA_EVALUABLE_FUNCTION_CUDA_H
#define LOCUSTA_EVALUABLE_FUNCTION_CUDA_H

#include "evaluable_function.hpp"
#include "math_constants.h"

namespace locusta {

    template<typename TFloat>
    struct evolutionary_solver_cuda;

    // template<typename TFloat>
    // __device__
    // void bound_mapping(const uint32_t Bound_Mapping_Method,
    //                    const TFloat &u,
    //                    const TFloat &l,
    //                    TFloat &x)
    // {
    //     if (x == x)
    //     {
    //         switch (Bound_Mapping_Method)
    //         {
    //         case 0: /// Out of bounds Crop
    //             x = x > u ? u : x < l ? l : x;
    //             break;
    //         case 1: /// Out of bounds Mirror
    //             x = x > u ? (2*u - x) : x < l ? (2*l - x) : x;
    //             break;
    //         case 2: /// Out of bounds Error
    //             if ( x > u || x < l )
    //             {
    //                 x = CUDART_NAN_F;
    //             }
    //             break;
    //         }
    //     }
    //     else
    //     {
    //         x = u;
    //     }
    //     return;
    // }

    template<typename TFloat>
    struct EvaluationCudaFunctor : EvaluationFunctor<TFloat>{
        virtual void operator()(evolutionary_solver_cuda<TFloat> * solver) {};
    };
}

#endif
