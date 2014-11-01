#ifndef LOCUSTA_PSO_OPERATORS_CUDA_H
#define LOCUSTA_PSO_OPERATORS_CUDA_H

#include <inttypes.h>

namespace locusta {

    template<typename TFloat>
    struct pso_solver_cuda;

    template<typename TFloat>
    struct UpdateSpeedCudaFunctor {
        virtual uint32_t required_prns(pso_solver_cuda<TFloat> * solver) {};
        virtual void operator()(pso_solver_cuda<TFloat> * solver) {};
    };

    template<typename TFloat>
    struct UpdateParticleRecordCudaFunctor {
        virtual uint32_t required_prns(pso_solver_cuda<TFloat> * solver) {};
        virtual void operator()(pso_solver_cuda<TFloat> * solver) {};
    };

    template<typename TFloat>
    struct UpdatePositionCudaFunctor {
        virtual uint32_t required_prns(pso_solver_cuda<TFloat> * solver) {};
        virtual void operator()(pso_solver_cuda<TFloat> * solver) {};
    };
}

#endif
