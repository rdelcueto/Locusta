#ifndef LOCUSTA_PSO_STD_OPERATORS_H
#define LOCUSTA_PSO_STD_OPERATORS_H

#include "pso_operators.hpp"

namespace locusta {

    template<typename TFloat>
    struct CanonicalSpeedUpdate : UpdateSpeedFunctor<TFloat> {
        void operator()(pso_solver<TFloat> * solver)
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

                const TFloat * prng_vector = const_cast<TFloat *>(solver->_bulk_prnumbers); // PRNGs vector

                for(uint32_t i = 0; i < ISLES; i++) {
                    const uint32_t isle_best_positions_offset = i * DIMENSIONS;
                    for(uint32_t j = 0; j < AGENTS; j++) {
                        const uint32_t particle_offset = i * AGENTS * DIMENSIONS + j * DIMENSIONS;
                        for(uint32_t k = 0; k < DIMENSIONS; k++) {
                            const uint32_t prng_offset = i * AGENTS * DIMENSIONS * 2 + j * DIMENSIONS * 2;
                            const TFloat c_rnd = prng_vector[prng_offset];
                            const TFloat s_rnd = prng_vector[prng_offset + 1];

                            const TFloat p_g = isle_best_positions[isle_best_positions_offset + k];
                            const TFloat p_i = particle_best_positions[particle_offset + k];
                                                const TFloat x_i = positions[particle_offset + k];

                            const TFloat v_i = velocities[particle_offset + k];
                            // Compute new velocity
                            velocities[particle_offset + k] =
                                inertia_factor * v_i +
                                cognitive_factor * c_rnd * (p_i - x_i) +
                                social_factor * s_rnd * (p_g - x_i);
                        }
                    }
                }
            }
    };

    template<typename TFloat>
    struct CanonicalPositionUpdate : UpdatePositionFunctor<TFloat> {
        void operator()(pso_solver<TFloat> * solver)
            {
                const uint32_t ISLES = solver->_ISLES;
                const uint32_t AGENTS = solver->_AGENTS;
                const uint32_t DIMENSIONS = solver->_DIMENSIONS;

                TFloat * new_positions = solver->_population->_transformed_data_array;
                const TFloat * positions = const_cast<TFloat *>(solver->_population->_data_array);
                const TFloat * velocities = const_cast<TFloat *>(solver->_velocity_vector);

                for(uint32_t i = 0; i < ISLES; i++) {
                    for(uint32_t j = 0; j < AGENTS; j++) {
                        const uint32_t genome_offset = i * AGENTS * DIMENSIONS + j * DIMENSIONS;
                        for(uint32_t k = 0; k < DIMENSIONS; k++) {
                            const uint32_t curr_gene = genome_offset + k;
                            const TFloat curr_velocity = velocities[curr_gene];
                            const TFloat curr_position = positions[curr_gene];

                            new_positions[curr_gene] = curr_position + curr_velocity;
                        }
                    }
                }
            }
    };
}

#endif
