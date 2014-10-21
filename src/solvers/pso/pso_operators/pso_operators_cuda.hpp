#ifndef LOCUSTA_PSO_OPERATORS_CUDA_H
#define LOCUSTA_PSO_OPERATORS_CUDA_H

namespace locusta {

    template<typename TFloat>
    struct pso_solver_cuda;

    template<typename TFloat>
    struct UpdateSpeedCudaFunctor {
        virtual void operator()(pso_solver_cuda<TFloat> * solver) {};
    };

    template<typename TFloat>
    struct UpdateParticleRecordCudaFunctor {
        virtual void operator()(pso_solver_cuda<TFloat> * solver) {};
    };

    template<typename TFloat>
    struct UpdatePositionCudaFunctor {
        virtual void operator()(pso_solver_cuda<TFloat> * solver) {};
    };
}

#endif
