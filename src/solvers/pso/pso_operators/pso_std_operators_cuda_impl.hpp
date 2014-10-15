#ifndef LOCUSTA_PSO_STD_OPERATORS_CUDA_H
#define LOCUSTA_PSO_STD_OPERATORS_CUDA_H

#include "pso_operators_cuda.hpp"

namespace locusta {

    template<typename TFloat>
    struct CanonicalSpeedCudaUpdate : UpdateSpeedCudaFunctor<TFloat> {
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
                TFloat * velocities = solver->_velocity_vector; // To be modified
                const TFloat * positions = const_cast<TFloat *>(solver->_population->_data_array); // Current
                // particle position.
                const TFloat * particle_best_positions = const_cast<TFloat *>(solver->_cognitive_position_vector); // (Cognitive)
                // Particle's
                // records
                const TFloat * isle_best_positions = const_cast<TFloat *>(solver->_best_genome); // (Social)
                // Isle's
                // records

                const TFloat * prng_vector = const_cast<TFloat *>(solver->_bulk_prnumbers); // PRNGs
                                                                                            // vector

                // TODO: canonical_speed_update_kernel dispatch.
            }
    };

    template<typename TFloat>
    struct CanonicalPositionCudaUpdate : UpdatePositionCudaFunctor<TFloat> {
        void operator()(pso_solver_cuda<TFloat> * solver)
            {
                const uint32_t ISLES = solver->_ISLES;
                const uint32_t AGENTS = solver->_AGENTS;
                const uint32_t DIMENSIONS = solver->_DIMENSIONS;

                TFloat * new_positions = solver->_population->_transformed_data_array;
                const TFloat * positions = const_cast<TFloat *>(solver->_population->_data_array);
                const TFloat * velocities = const_cast<TFloat *>(solver->_velocity_vector);

                // TODO: canonical_position_update_kernel dispatch
            }
    };
}

#endif
