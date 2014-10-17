#ifndef LOCUSTA_PSO_STD_OPERATORS_CUDA_H
#define LOCUSTA_PSO_STD_OPERATORS_CUDA_H

#include "pso_operators_cuda.hpp"

namespace locusta {

    template<typename TFloat>
    void canonical_speed_update_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const TFloat inertia_factor,
     const TFloat cognitive_factor,
     const TFloat social_factor,
     const TFloat * positions,
     const TFloat * best_positions,
     const TFloat * isle_best_positions,
     const TFloat * prng_vector,
     TFloat * velocities);

    template <typename TFloat>
    void canonical_position_update_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const TFloat * velocities,
     const TFloat * positions,
     TFloat * new_positions);

    template<typename TFloat>
    struct CanonicalSpeedUpdateCuda : UpdateSpeedCudaFunctor<TFloat> {
        void operator()(pso_solver_cuda<TFloat> * solver)
            {
                const uint32_t ISLES = solver->_ISLES;
                const uint32_t AGENTS = solver->_AGENTS;
                const uint32_t DIMENSIONS = solver->_DIMENSIONS;

                // Solver configuration constants
                const TFloat inertia_factor = solver->_inertia_factor;
                const TFloat cognitive_factor = solver->_cognitive_factor;
                const TFloat social_factor = solver->_social_factor;

                // Solver state
                const TFloat * positions = const_cast<TFloat *>(solver->_dev_population->_dev_data_array); // Current
                // particle position.
                const TFloat * best_positions = const_cast<TFloat *>(solver->_dev_cognitive_position_vector); // (Cognitive)
                // Particle's
                // records
                const TFloat * isle_best_positions = const_cast<TFloat *>(solver->_dev_best_genome); // (Social)
                // Isle's
                // records
                const TFloat * prng_vector = const_cast<TFloat *>(solver->_dev_bulk_prnumbers); // PRNGs

                TFloat * velocities = solver->_dev_velocity_vector; // To be
                                                                    // modified

                canonical_speed_update_dispatch(ISLES,
                                                AGENTS,
                                                DIMENSIONS,
                                                inertia_factor,
                                                cognitive_factor,
                                                social_factor,
                                                positions,
                                                best_positions,
                                                isle_best_positions,
                                                prng_vector,
                                                velocities);

            }
    };

    template<typename TFloat>
    struct CanonicalPositionUpdateCuda : UpdatePositionCudaFunctor<TFloat> {
        void operator()(pso_solver_cuda<TFloat> * solver)
            {
                const uint32_t ISLES = solver->_ISLES;
                const uint32_t AGENTS = solver->_AGENTS;
                const uint32_t DIMENSIONS = solver->_DIMENSIONS;

                const TFloat * velocities = const_cast<TFloat *>(solver->_dev_velocity_vector);
                const TFloat * positions = const_cast<TFloat *>(solver->_dev_population->_dev_data_array);

                TFloat * new_positions = solver->_dev_population->_dev_transformed_data_array;

                canonical_position_update_dispatch(ISLES,
                                                   AGENTS,
                                                   DIMENSIONS,
                                                   velocities,
                                                   positions,
                                                   new_positions);
            }
    };
}

#endif
